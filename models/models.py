
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single' or opt.dataset_mode == 'singlemat' or opt.dataset_mode == 'mat')
        from .test_model import TestModel
        model = TestModel()
    elif opt.model == 'depth_pix2pix':
        assert(opt.dataset_mode == 'mat')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'cnn_depth':
        assert(opt.dataset_mode == 'superpix')
        from .cnn_depth_model import CNNDepthModel
        model = CNNDepthModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
