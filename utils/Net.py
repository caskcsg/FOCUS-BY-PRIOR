import torch

def remove_prefix(state_dict, prefix):
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, device=0):
    print('Loading pretrained model from {}'.format(pretrained_path))
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    pretrained_dict = remove_prefix(pretrained_dict['model'], 'module.')
    #print (pretrained_dict.keys())
    model.load_state_dict(pretrained_dict, strict=False)
    del pretrained_dict
    torch.cuda.empty_cache()
    return model
    
    
def load_weights(model, checkpoint_path, multi_gpu):
    print('Loading checkpoint ', checkpoint_path)
    # checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(gpu_id))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if multi_gpu:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint['model'], strict=False)
    del checkpoint
    torch.cuda.empty_cache()
    return model

def load_pretrained(model, path_param, device=None, prex='.'):
    #path_param = prex + '/cse_s_data/checkpoints_final/' + path_param
    checkpoint = torch.load(path_param, map_location='cpu')
    pretrained_dict = checkpoint['model']
    # pretrained_dict = torch.load(path_param)
    model_dict = model.state_dict()
    pretrained_dict_trans = {}
    for k, v in pretrained_dict.items():
        if 'module' in k and k[7:] in model_dict.keys():
            pretrained_dict_trans[k[7:]] = v
        else:
            pretrained_dict_trans[k] = v
    model_dict.update(pretrained_dict_trans)
    model.load_state_dict(model_dict)
   # if optim is not None:
    #    # checkpoint = torch.load(path_param, map_location=lambda storage, loc: storage)
        # print(checkpoint['optimizer_state_dict'])
   #     optim.load_state_dict(checkpoint['optimizer_state_dict'])
     #   for state in optim.state.values():
           # for k, v in state.items():
               # if torch.is_tensor(v):
                  #  state[k] = v.cuda()
    #model.eval()
    return model