import meta_wmnet as wmnet
import torch
device = torch.device("cpu")
def get_wm(folder_name, model_idx, wm_loss=None, n_frames=4): # 'mse', 'bilinear', 'l2'
        knowledge_size_hidden_layers = [256,128]
        param_size_hidden_layers = [256,128]
        prediction_size_hidden_layers = [128,64]
        projection_size_hidden_layers = [64,32]
        knowledge_latent_dim = 64
        param_latent_dim = 32
        projection_latent_dim = 16        
        state_dim = 16 * n_frames
        knowledge_num_hidden_layers = len(knowledge_size_hidden_layers)
        param_num_hidden_layers = len(param_size_hidden_layers)
        prediction_num_hidden_layers = len(prediction_size_hidden_layers)
        projection_num_hidden_layers = len(projection_size_hidden_layers)
        # Each frame contain
        player_state_dim = 4
        ball_state_dim = 4
        ball_player_int_dim = 1
        ball_map_int_dim = 5


        knowledge_net_input_dim =  ball_state_dim * n_frames
        knowledge_net_output_dim = knowledge_latent_dim

        param_net_input_dim = ball_state_dim * n_frames
        param_net_output_dim = param_latent_dim

        prediction_net_input_dim = knowledge_net_output_dim + param_net_output_dim
        prediction_net_output_dim = ball_state_dim

        projection_net_input_dim = param_net_output_dim
        projection_net_output_dim = projection_latent_dim

        knowledge_net = wmnet.knowledge_net(knowledge_net_input_dim, knowledge_net_output_dim, 
                                            knowledge_num_hidden_layers,
                                            knowledge_size_hidden_layers).to(device)


        # NN that trying to extract physical parameters
        param_net = wmnet.param_net(param_net_input_dim, param_net_output_dim, 
                                  param_num_hidden_layers, param_size_hidden_layers).to(device)

        prediction_net = wmnet.prediction_net(prediction_net_input_dim, prediction_net_output_dim, 
                                  prediction_num_hidden_layers, prediction_size_hidden_layers).to(device)

        projection_net = wmnet.projection_net(projection_net_input_dim, projection_net_output_dim, 
                                  projection_num_hidden_layers, projection_size_hidden_layers).to(device)    
        
        if wm_loss == 'bilinear':
            prefix = folder_name + "wm_con_"
        elif wm_loss == 'l2':
            prefix = folder_name + "wm_con_l2_"
        else:
            prefix = folder_name + "wm_"
            
        prefix2 = prefix + str(model_idx) + "_"
        
        knowledge_net.load_state_dict(torch.load(prefix2+ "knowledge_net.pth"))
        param_net.load_state_dict(torch.load(prefix2+ "param_net.pth"))
        prediction_net.load_state_dict(torch.load(prefix2+ "prediction_net.pth"))

        knowledge_net.eval()
        param_net.eval()
        prediction_net.eval()
        
        wm_nets = {'knowledge_net':knowledge_net, 'param_net': param_net, 
                   'prediction_net':prediction_net, 'projection_net':projection_net}
        
        return wm_nets