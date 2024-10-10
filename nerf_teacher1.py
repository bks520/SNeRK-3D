from run_nerf import  *
import torch.nn.functional as F
from load_llff1 import *

def render_path3(render_poses, hwf, K, chunk, render_kwargs, gt_imgs, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []


    c2w = render_poses[0]
    rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)


    resized_tensor = F.adaptive_max_pool2d(rgb.permute(2, 0, 1).unsqueeze(0), (1024, 1))
    resized_tensor = resized_tensor.squeeze(0).permute(1, 2, 0)
    resized_tensor = resized_tensor.squeeze(1)

    

    rgbs.append(rgb.cpu().numpy())
    disps.append(disp.cpu().numpy())
    i = 0

    if i==0:
        rgbd = resized_tensor

    if savedir is not None:
        rgb8 = to8b(rgbs[-1])
        filename = os.path.join(savedir, '{:03d}.png'.format(i))
        imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps, rgbd



def create_nerf1(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'ta' in f]

#     print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
#         print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################
    # 'network_fn' : model：将名为model的变量作为值与键'network_fn'关联。
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
#         print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def teacher1(target_s):

    parser = config_parser()
    # 使用 parser.parse_args() 解析命令行参数，并将结果保存在 args 变量中。
    args = parser.parse_args()


    K = None

    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data1(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)

        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]

        if not isinstance(i_test, list):
            i_test = [i_test]
        if args.llffhold > 0:

            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test

        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])


        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    basedir = args.basedir
    expname = args.expname

    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf1(args)
    # global_step = start
    start =100000

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)



    with torch.no_grad():
        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)

        rgbs, _ , rgbd= render_path3(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=target_s, savedir=testsavedir, render_factor=args.render_factor)

        img_loss = img2mse(rgbd, target_s)

        psnr = img_loss

        return psnr
    
if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    teacher1()