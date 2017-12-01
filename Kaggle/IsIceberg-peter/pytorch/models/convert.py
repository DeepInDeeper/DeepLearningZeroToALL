import numpy as np
import torch



# Convert the np arrays into the correct dimention and type
# Note that BCEloss requires Float in X as well as in y
def XnumpyToTensor(x_data_np):
    x_data_np = np.array(x_data_np, dtype=np.float32)
    use_cuda = 1
    if use_cuda:
        # lgr.info("Using the GPU")
        X_tensor = (torch.from_numpy(x_data_np).cuda())  # Note the conversion for pytorch
    else:
        # lgr.info("Using the CPU")
        X_tensor = (torch.from_numpy(x_data_np))  # Note the conversion for pytorch
    return X_tensor


# Convert the np arrays into the correct dimention and type
# Note that BCEloss requires Float in X as well as in y
def YnumpyToTensor(y_data_np):
    y_data_np = y_data_np.reshape((y_data_np.shape[0], 1))  # Must be reshaped for PyTorch!
    use_cuda = 1
    if use_cuda:
        # lgr.info("Using the GPU")
        #     Y = Variable(torch.from_numpy(y_data_np).type(torch.LongTensor).cuda())
        Y_tensor = (torch.from_numpy(y_data_np)).type(torch.FloatTensor).cuda()  # BCEloss requires Float
    else:
        # lgr.info("Using the CPU")
        #     Y = Variable(torch.squeeze (torch.from_numpy(y_data_np).type(torch.LongTensor)))  #
        Y_tensor = (torch.from_numpy(y_data_np)).type(torch.FloatTensor)  # BCEloss requires Float
    return Y_tensor
