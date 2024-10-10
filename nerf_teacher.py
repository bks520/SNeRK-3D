from run_nerf import  *
import torch.nn.functional as F
from load_llff1 import *



def render_path1(render_poses, hwf, K, chunk, render_kwargs, gt_imgs, savedir=None, render_factor=0):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []


    for i, c2w in enumerate(tqdm(render_poses)):
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)

        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
#         psnr = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs)))
        if i == 0:
            rgbd = rgb
            print(rgb.shape, disp.shape)
            print('rgbd shape:', rgbd.shape)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps, rgbd

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
#     print('rgb shape:', rgb.shape)


    resized_tensor = F.adaptive_max_pool2d(rgb.permute(2, 0, 1).unsqueeze(0), (1024, 1))
    resized_tensor = resized_tensor.squeeze(0).permute(1, 2, 0)
    resized_tensor = resized_tensor.squeeze(1)
#     print(resized_tensor.shape)
    

    rgbs.append(rgb.cpu().numpy())
    disps.append(disp.cpu().numpy())
    i = 0

    if i==0:
        rgbd = resized_tensor
#         print(rgb.shape, disp.shape)


#     if gt_imgs is not None and render_factor==0:
#         p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
#         psnr = img2mse(rgb, gt_imgs[i])
#         print(p)
#         print('psnr2',psnr)


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

def teacher(target_s):

    parser = config_parser()
    # 使用 parser.parse_args() 解析命令行参数，并将结果保存在 args 变量中。
    args = parser.parse_args()

    # Load data
    K = None
    # 解析了配置参数，然后根据数据集类型加载数据。根据不同的数据集类型，加载对应的数据集和相关参数。
    # 根据 args.dataset_type 的值来确定数据集的类型，并加载相应的数据集和相关参数
    # 对于 llff 类型的数据集，加载了图像、相机姿态、边界值等数据。
    # 该函数接受一些参数，
    # 如数据目录 args.datadir、缩放因子 args.factor、是否重新调整图像中心 recenter、边界因子 bd_factor 和是否球面化 args.spherify。
    # 加载的数据包括图像、相机姿态、边界值、渲染相机姿态和测试图像索引。
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data1(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        # hwf 表示相机的高度、宽度和焦距，从姿态数据中获取。
        hwf = poses[0, :3, -1]
        # 对姿态数据进行处理，保留前三行（旋转矩阵）和前四列（平移向量）。
        poses = poses[:, :3, :4]
        # 打印加载的 LLFF 数据集的形状、渲染相机姿态的形状、相机参数 hwf 和数据目录。
#         print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        # 如果 i_test 不是列表类型，则将其转换为列表。
        if not isinstance(i_test, list):
            i_test = [i_test]
        # 如果 args.llffhold 大于 0，则根据 args.llffhold 的值自动生成测试集索引 i_test。
        if args.llffhold > 0:
#             print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]
        # 将 i_test 设置为验证集索引 i_val。
        i_val = i_test
        # 根据图像索引 i_test 和 i_val，计算训练集索引 i_train。
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

#         print('DEFINING BOUNDS')
        # 定义边界值（near 和 far）。
        # 如果 args.no_ndc 为 True，则根据边界值数组 bds 计算边界值的估计值（near 和 far）。
        # 如果 args.no_ndc 为 False，则将 near 设置为 0，将 far 设置为 1。
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        # 打印边界值的估计结果
#         print('NEAR FAR', near, far)

    # 对于 blender 类型的数据集，加载了图像、相机姿态、渲染相机姿态、相机参数等数据。
    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
#         print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    # 对于 LINEMOD 类型的数据集，加载了图像、相机姿态、渲染相机姿态、相机参数等数据。
    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res,
                                                                                    args.testskip)
#         print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
#         print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    # 对于 deepvoxels 类型的数据集，加载了图像、相机姿态、渲染相机姿态、相机参数等数据。
    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

#         print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.
        far = hemi_R + 1.

    else:
#         print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
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

    # Short circuit if only rendering out from trained model

    with torch.no_grad():
        # if args.render_test:
        #     # render_test switches to test poses
        #     images = images[i_test]
        # else:
        #     # Default is smoother render_poses path
        #     images = None

        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)


        rgbs, _ , rgbd= render_path3(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=target_s, savedir=testsavedir, render_factor=args.render_factor)

        img_loss = img2mse(rgbd, target_s)
#         print('psnr', img_loss)
        psnr = img_loss


#         print('Done rendering', testsavedir)
#         imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

        return psnr
    
if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    teacher()