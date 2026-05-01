import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    mt = beta1 * np.array(m) + (1 - beta1) * np.array(grad)
    vt = beta2 * np.array(v) + (1 - beta2) * np.array(grad)**2
    bias_mt = np.array(mt) / (1 - beta1**t)
    bias_vt = np.array(vt) / (1 - beta2**t)
    paramt = param - lr * (bias_mt / (np.sqrt(bias_vt) + eps))
    return paramt, mt, vt