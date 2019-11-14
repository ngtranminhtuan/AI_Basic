import math
import numpy as np
import matplotlib.pyplot as plt

X0 = np.array([[-0.00000000e+00, 0.00000000e+00], [1.02370162e-03, 1.00490019e-02], [4.46886653e-04, 2.01970768e-02],
               [1.47407793e-02, 2.64760849e-02], [4.89297702e-03, 4.01066735e-02],
               [2.24532697e-02, 4.52394828e-02], [-1.29046709e-02, 5.92162482e-02], [1.40119290e-02, 6.93048028e-02],
               [2.81378611e-02, 7.57509518e-02], [3.10420311e-02, 8.54450415e-02],
               [1.25289291e-02, 1.00230068e-01], [4.98602596e-02, 9.92956873e-02], [5.98467626e-02, 1.05407511e-01],
               [7.44294163e-02, 1.08182256e-01], [8.10938613e-02, 1.15852255e-01],
               [4.77308594e-02, 1.43800578e-01], [7.49463712e-02, 1.43188076e-01], [6.99562028e-02, 1.56821289e-01],
               [8.52544719e-02, 1.60591177e-01], [1.04317824e-01, 1.61092420e-01],
               [1.24506678e-01, 1.59091952e-01], [1.46850999e-01, 1.53069242e-01], [1.98176961e-01, 1.00541574e-01],
               [2.21210027e-01, 7.09944241e-02], [2.08869701e-01, 1.23056741e-01],
               [2.44866157e-01, 6.17217021e-02], [2.62330456e-01, 1.24613711e-02], [2.59597194e-01, 8.36030034e-02],
               [2.63026068e-01, 1.03966942e-01], [2.86274023e-01, 6.20866672e-02],
               [2.71990699e-01, 1.33597995e-01], [3.06572144e-01, 6.37553095e-02], [2.99701037e-01, 1.21071974e-01],
               [2.86705335e-01, 1.70032827e-01], [3.26457692e-01, 1.06642036e-01],
               [3.45419731e-01, 7.53157072e-02], [3.61673246e-01, -3.77341786e-02], [3.73725607e-01, 2.96564652e-03],
               [3.83776441e-01, -6.89550785e-03], [3.60223698e-01, -1.59458878e-01],
               [4.03721926e-01, -1.60391465e-02], [3.97377250e-01, -1.16638037e-01], [4.05557799e-01, -1.24517092e-01],
               [4.25053905e-01, -8.93498569e-02], [4.42919300e-01, 3.67880116e-02],
               [4.18978232e-01, -1.76263471e-01], [4.60163483e-01, -6.43887118e-02], [4.25487584e-01, -2.10583667e-01],
               [3.47537283e-01, -3.38076752e-01], [4.91581606e-01, -5.76413695e-02],
               [4.38156190e-01, -2.51187511e-01], [4.48456616e-01, -2.53510842e-01], [4.40495398e-01, -2.86101414e-01],
               [4.13063834e-01, -3.40560826e-01], [5.21612338e-01, -1.59503071e-01],
               [5.10516450e-01, -2.19123093e-01], [4.57177299e-01, -3.33100987e-01], [4.29663032e-01, -3.83257700e-01],
               [3.33715611e-01, -4.81522766e-01], [2.99530882e-01, -5.15217518e-01],
               [2.93478125e-01, -5.30264131e-01], [3.87797007e-01, -4.78820027e-01], [3.64858997e-01, -5.09001758e-01],
               [4.62058507e-01, -4.37562125e-01], [4.52750809e-01, -4.61446903e-01],
               [3.28120262e-01, -5.68696365e-01], [3.55850725e-01, -5.63750571e-01], [2.35008863e-01, -6.34653703e-01],
               [3.05228971e-01, -6.15324198e-01], [2.66094288e-01, -6.44174346e-01],
               [1.91280385e-01, -6.80706103e-01], [1.53058548e-01, -7.00648523e-01], [1.27296729e-01, -7.16045503e-01],
               [-1.97987361e-02, -7.37107888e-01], [1.94217072e-01, -7.21802069e-01],
               [1.02253038e-01, -7.50643287e-01], [5.46017367e-02, -7.65732506e-01], [-1.35364339e-01, -7.65907806e-01],
               [-1.47883663e-02, -7.87739988e-01], [-1.23753238e-01, -7.88325373e-01],
               [6.69563082e-02, -8.05302083e-01], [-1.28530457e-01, -8.08023149e-01],
               [-2.33431171e-01, -7.94708961e-01], [-9.41407684e-02, -8.33081614e-01],
               [-1.09877654e-01, -8.41340264e-01],
               [-2.09543434e-01, -8.32623100e-01], [-3.87572025e-01, -7.77434757e-01],
               [-3.61471707e-01, -8.01003334e-01], [-1.89827604e-01, -8.68382944e-01],
               [-3.79272267e-01, -8.15067719e-01],
               [-1.82125481e-01, -8.90660761e-01], [-3.36907291e-01, -8.55223515e-01],
               [-6.06080034e-01, -7.04451802e-01], [-5.36755429e-01, -7.70943955e-01],
               [-5.90315883e-01, -7.43685295e-01],
               [-5.50402765e-01, -7.86054198e-01], [-6.12833322e-01, -7.51496862e-01],
               [-5.24221081e-01, -8.27765872e-01], [-4.28940594e-01, -8.92137869e-01],
               [-6.64081054e-01, -7.47660588e-01]])
X1 = np.array(
    [[-0.00000000e+00, -0.00000000e+00], [-8.90404749e-03, -4.76952234e-03], [-1.73009287e-02, -1.04306993e-02],
     [-2.11171063e-02, -2.17334183e-02], [-3.13247799e-02, -2.55194953e-02],
     [-4.93486085e-02, -1.07459279e-02], [-2.97582243e-02, -5.27971843e-02], [-6.45660642e-02, -2.88220957e-02],
     [-7.88086943e-02, -1.78643675e-02], [-8.55769020e-02, -3.06766466e-02],
     [-9.07250300e-02, -4.44073129e-02], [-1.03684637e-01, -3.99396424e-02], [-1.06612405e-01, -5.76729865e-02],
     [-1.11580459e-01, -6.92310600e-02], [-1.41414104e-01, -1.03284724e-04],
     [-1.49823741e-01, -2.25762638e-02], [-1.49364003e-01, -6.17266408e-02], [-1.63779017e-01, -5.16064013e-02],
     [-1.81817155e-01, 6.11080795e-04], [-1.88637721e-01, 3.53381742e-02],
     [-2.02020201e-01, -1.46928612e-05], [-2.06879590e-01, 4.68641014e-02], [-2.21070844e-01, 2.25919899e-02],
     [-2.28965936e-01, 3.93533254e-02], [-2.35169101e-01, 5.88643111e-02],
     [-2.48101809e-01, 4.70584279e-02], [-2.61315833e-01, -2.62028476e-02], [-2.23298428e-01, 1.56582174e-01],
     [-2.72808565e-01, 7.46145058e-02], [-2.68553284e-01, 1.16990189e-01],
     [-2.47127942e-01, 1.75371449e-01], [-2.53825721e-01, 1.83367725e-01], [-2.15435032e-01, 2.40970707e-01],
     [-2.16048009e-01, 2.53839258e-01], [-2.57219938e-01, 2.27563292e-01],
     [-2.54316994e-01, 2.45581174e-01], [-2.96706237e-01, 2.10230383e-01], [-2.10863840e-01, 3.08571006e-01],
     [-3.41040093e-02, 3.82320313e-01], [-2.32277951e-01, 3.18174794e-01],
     [-1.33437491e-01, 3.81370009e-01], [-2.82301944e-01, 3.03016045e-01], [-2.81586362e-01, 3.17318066e-01],
     [-2.44199374e-01, 3.59194773e-01], [-1.07997512e-01, 4.31123418e-01],
     [-2.40098234e-01, 3.85959076e-01], [-3.22117875e-02, 4.63528573e-01], [-1.63340991e-01, 4.45763262e-01],
     [-3.34883188e-02, 4.83690589e-01], [-2.00997505e-01, 4.52299685e-01],
     [-1.05342850e-01, 4.93942200e-01], [-1.46665110e-01, 4.93832390e-01], [-1.17597074e-02, 5.25120867e-01],
     [2.06003182e-02, 5.34957040e-01], [-1.76097763e-01, 5.16246297e-01],
     [3.00571829e-02, 5.54741869e-01], [-1.82423719e-01, 5.35433410e-01], [-6.25071555e-02, 5.72354472e-01],
     [-8.50801034e-02, 5.79647875e-01], [2.88777690e-01, 5.21320713e-01],
     [1.87045308e-01, 5.76475074e-01], [-1.24089441e-01, 6.03537031e-01], [3.17995327e-01, 5.39521871e-01],
     [8.27815865e-02, 6.30956327e-01], [2.52419165e-01, 5.95147968e-01],
     [3.74641582e-01, 5.39186560e-01], [4.63715951e-01, 4.78969687e-01], [2.48555860e-01, 6.29471582e-01],
     [3.74077052e-01, 5.76068531e-01], [1.55452491e-01, 6.79412453e-01],
     [3.81178454e-01, 5.95526633e-01], [2.33774063e-01, 6.78000708e-01], [3.26359778e-01, 6.49934547e-01],
     [3.37805823e-01, 6.55444319e-01], [5.50516869e-01, 5.05618112e-01],
     [4.17191288e-01, 6.32354693e-01], [5.41695656e-01, 5.43960877e-01], [6.44362978e-01, 4.35585381e-01],
     [4.75526138e-01, 6.28194139e-01], [5.52243922e-01, 5.76019452e-01],
     [7.25356478e-01, 3.56163687e-01], [7.30444910e-01, 3.68607814e-01], [7.03710683e-01, 4.36856634e-01],
     [6.56144348e-01, 5.21883181e-01], [8.48428181e-01, 9.80611659e-03],
     [6.36553778e-01, 5.76167480e-01], [7.07730818e-01, 5.03720125e-01], [8.11587056e-01, 3.37037960e-01],
     [8.81451899e-01, 1.14743220e-01], [8.86410966e-01, 1.49861395e-01],
     [8.50996760e-01, 3.19766782e-01], [8.13538050e-01, 4.27866363e-01], [9.18695781e-01, 1.39940737e-01],
     [9.39374941e-01, -5.97437662e-03], [9.46227513e-01, 7.87029352e-02],
     [9.41703213e-01, 1.84443664e-01], [9.69634480e-01, 1.10085579e-02], [9.64491863e-01, -1.72509500e-01],
     [9.08099555e-01, -3.94024376e-01], [9.83692635e-01, 1.79857721e-01]])
X2 = np.array([[0.00000000e+00, -0.00000000e+00], [1.00796294e-02, 6.56867843e-04], [1.99627786e-02, -3.09985355e-03],
               [2.97027062e-02, -6.00190720e-03], [3.89803692e-02, -1.06309594e-02],
               [4.60802930e-02, -2.06728500e-02], [4.68979393e-02, -3.83885123e-02], [6.18421657e-02, -3.42788038e-02],
               [7.79294361e-02, -2.13763633e-02], [8.98579213e-02, -1.37846577e-02],
               [1.00985582e-01, 2.22547817e-03], [9.44630762e-02, -5.85013354e-02], [9.15146721e-02, -7.94823447e-02],
               [8.54520349e-02, -9.97050058e-02], [1.12272693e-01, -8.59814040e-02],
               [5.81500133e-02, -1.39912176e-01], [1.41248688e-01, -7.85403839e-02], [1.21355094e-01, -1.21489622e-01],
               [7.38684496e-02, -1.66136400e-01], [1.00106472e-01, -1.63742696e-01],
               [1.53044250e-01, -1.31869707e-01], [1.16760582e-01, -1.77094255e-01], [6.12276871e-02, -2.13620894e-01],
               [6.73877706e-02, -2.22335271e-01], [6.43916089e-02, -2.33716140e-01],
               [1.50474080e-01, -2.02796830e-01], [-4.54224108e-02, -2.58668433e-01], [1.03267291e-01, -2.52420348e-01],
               [5.19627451e-02, -2.78013868e-01], [2.89150022e-02, -2.91498702e-01],
               [-1.60303799e-04, -3.03030261e-01], [3.30973975e-02, -3.11377233e-01], [4.33978502e-02, -3.20305731e-01],
               [-1.55979933e-02, -3.32968187e-01], [7.76266595e-02, -3.34546335e-01],
               [-2.11057939e-02, -3.52904791e-01], [-1.11850913e-01, -3.46006905e-01],
               [1.59000191e-01, -3.38228567e-01], [-2.87394749e-02, -3.82760953e-01], [7.29470900e-02, -3.87126553e-01],
               [-1.63193605e-01, -3.69616687e-01], [-9.58943467e-02, -4.02886318e-01],
               [-1.50983618e-01, -3.96466369e-01], [-1.77264626e-01, -3.96524238e-01],
               [-9.79622123e-02, -4.33513863e-01],
               [-1.15089003e-01, -4.39734115e-01], [-2.43344709e-01, -3.95827853e-01],
               [-2.33948737e-01, -4.13101868e-01], [-2.56227899e-01, -4.11613067e-01],
               [-1.95315739e-01, -4.54782107e-01],
               [-3.89323660e-01, -3.21718978e-01], [-3.67964982e-01, -3.60531352e-01],
               [-2.19271679e-01, -4.77294611e-01], [-4.18578956e-01, -3.33758993e-01],
               [-2.71453008e-01, -4.73110902e-01],
               [-2.94309262e-01, -4.71194263e-01], [-4.25602910e-01, -3.72598327e-01],
               [-4.60473008e-01, -3.45631877e-01], [-1.53035431e-01, -5.65517851e-01],
               [-5.38525136e-01, -2.55261665e-01],
               [-5.75133781e-01, -1.91129779e-01], [-5.45489577e-01, -2.86524448e-01],
               [-5.70161066e-01, -2.59077663e-01], [-6.00620476e-01, -2.10270592e-01],
               [-5.83286958e-01, -2.78734397e-01],
               [-6.18434866e-01, -2.20492127e-01], [-5.64072143e-01, -3.55340768e-01],
               [-6.64463808e-01, -1.28461421e-01], [-6.09444281e-01, -3.16806347e-01],
               [-6.94594303e-01, -5.74935910e-02],
               [-6.91083080e-01, -1.49509739e-01], [-7.16715236e-01, 2.55840365e-02],
               [-6.33336314e-01, -3.57506270e-01], [-7.25653628e-01, 1.30945947e-01], [-6.94257452e-01, 2.76992940e-01],
               [-7.36721061e-01, -1.76530755e-01], [-7.67582571e-01, 1.20256522e-02], [-7.46063671e-01, 2.19834644e-01],
               [-7.79618854e-01, 1.13786763e-01], [-7.78796908e-01, 1.73917034e-01],
               [-8.07016720e-01, 4.14560690e-02], [-7.16468498e-01, 3.95087812e-01], [-8.28267121e-01, 5.10098662e-03],
               [-8.11507877e-01, 2.10576416e-01], [-7.33316503e-01, 4.26817812e-01],
               [-7.53378075e-01, 4.11814463e-01], [-8.52204966e-01, 1.68414878e-01], [-8.71983844e-01, 1.09143538e-01],
               [-8.33154605e-01, 3.09801325e-01], [-5.89949780e-01, 6.78337745e-01],
               [-4.40217251e-01, 7.95396161e-01], [-4.96992675e-01, 7.73247739e-01], [-7.23489625e-01, 5.83222180e-01],
               [-7.19137960e-01, 6.04401826e-01], [-5.73739592e-01, 7.56547117e-01],
               [-2.82807174e-01, 9.16975740e-01], [-3.23567812e-01, 9.14120388e-01], [-5.57559214e-01, 8.05687163e-01],
               [-4.94896995e-01, 8.57307981e-01], [-5.96268801e-01, 8.02784851e-01]])

X = np.concatenate((X0, X1, X2), axis=0)
y = np.concatenate(np.array([[0] * len(X0), [1] * len(X1), [2] * len(X2)]), axis=0).T


#####################################################################################

def calculate_cross_entropy(y, y_hat):
    return -np.sum(y * np.log(y_hat))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


# V: size n*c
#    n is the number of one-hot coding labels corresponding to provided sample X
#    c is the number of total label
def softmax(V):
    # TODO: implement softmax function
    e_x = np.exp(V - np.max(V))
    return e_x / e_x.sum(axis=0)


# V: size m*n for any interger m, n
def reLU(V):
    # TODO: set every negative element in matrix V to zero
    return V * (V > 0)


# V: size m*n for any interger m, n
def reLU_grad(V):
    # TODO: set every positive element in matrix V to 1
    # otherwise: set to 0
    return 1. * (V > 0)


## One-hot coding
# y: size n*1. Is an array of labels corresponding to provided sample X
# C: total labels
def generate_onehot_coding(y, C=3):
    # TODO: Convert every label y_i to corresponding one_hot coding array
    # y_i=0 ==> one_hot_coding = [1, 0, 0]
    # y_i=1 ==> one_hot_coding = [0, 1, 0]
    # y_i=2 ==> one_hot_coding = [0, 0, 1]
    targets = np.array(y).reshape(-1)
    return np.eye(C)[targets]


# X: size n*f
#    n is the number of sample
#    f is the number of feature of single sample
def generate_X_dot(X):
    # TODO: Append a column of number 1 to the left of X
    return np.concatenate((np.ones((1, X.shape[0])).T, X), axis=1)


# cost or loss function
# https://en.wikipedia.org/wiki/Cross_entropy
# Y:     matrix size n*f
#        n is the number of sample
#        f is the length of onehot coding
# Y_hat: matrix size n*f, as same as Y

# H(y,p)=−∑iyilog(pi)
def cost(Y, Y_hat):
    n = Y.shape[0]

    result = 0
    for i in range(n):
        ce = calculate_cross_entropy(Y[i], Y_hat[i])
        result = result + ce

    return result


def feed_forward(X_dot, W1_dot, W2_dot, activationFunction):
    # TODO:
    # Calculate Z1
    # Calculate A1
    # Calculate A1_dot
    # Calculate Z2
    # Calculate Y hat
    # return Z1, A1_dot, Z2, Y_hat
    pass

def predict_class(X_dot, W1_dot, W2_dot, activationFunction):
  Z1, A1_dot, Z2, Y_hat = feed_forward(X_dot, W1_dot, W2_dot, activationFunction)
  return np.argmax(Y_hat, axis=1)
