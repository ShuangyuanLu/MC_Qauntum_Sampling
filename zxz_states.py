from MPS_basic import MPS_basic_spin_one_half
import numpy as np


def build_zxz_state_and_compute_corr(L=20, x=0.03):
    # L = 20 #even
    state_mps = []
    for site in range(L):
        state_mps.append(np.ones([2, 1, 1])/np.sqrt(2))

    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    id = np.eye(2)

    CZ_left = np.zeros([2, 2, 2])
    CZ_left[:, :, 0] = (id + Z) /2
    CZ_left[:, :, 1] = (id - Z) /2
    CZ_right = np.zeros([2, 2, 2])
    CZ_right[:, :, 0] = id
    CZ_right[:, :, 1] = Z
    for site in range(L -1):
        tmp = np.einsum("yab,xyc->xabc", state_mps[site], CZ_left)
        state_mps[site] = tmp.reshape(tmp.shape[0], tmp.shape[1], tmp.shape[2] * tmp.shape[3])
    for site in range(1, L):
        tmp = np.einsum("yab,xyc->xacb", state_mps[site], CZ_right)
        state_mps[site] = tmp.reshape(tmp.shape[0], tmp.shape[1] * tmp.shape[2], tmp.shape[3])

    # x = 0.03
    theta_ZZ = x * np.pi
    theta_X = x * np.pi
    theta_XX = x * np.pi

    XX_left = np.zeros([2, 2, 2], dtype=np.complex128)
    XX_left[:, :, 0] = np.cos(theta_XX) * id
    XX_left[:, :, 1] = 1j* np.sin(theta_XX) * X
    XX_right = np.zeros([2, 2, 2], dtype=np.complex128)
    XX_right[:, :, 0] = id
    XX_right[:, :, 1] = X

    for site in range(L-1):
        tmp = np.einsum("yab,xyc->xabc", state_mps[site], XX_left)
        state_mps[site] = tmp.reshape(tmp.shape[0], tmp.shape[1], tmp.shape[2] * tmp.shape[3])
    for site in range(1, L):
        tmp = np.einsum("yab,xyc->xacb", state_mps[site], XX_right)
        state_mps[site] = tmp.reshape(tmp.shape[0], tmp.shape[1] * tmp.shape[2], tmp.shape[3])


    Ux = np.cos(theta_X) * id + 1j * np.sin(theta_X) * X
    ZZ_left = np.zeros([2, 2, 2], dtype=np.complex128)
    ZZ_left[:, :, 0] = np.cos(theta_ZZ) * id @ Ux
    ZZ_left[:, :, 1] = 1j* np.sin(theta_ZZ) * Z @ Ux
    ZZ_right = np.zeros([2, 2, 2], dtype=np.complex128)
    ZZ_right[:, :, 0] = id @ Ux
    ZZ_right[:, :, 1] = Z @ Ux
    ZZ_mid= id

    # even site ZZ
    for i in range(0, L // 2 - 1, 2):
        site = 2 * i
        tmp = np.einsum("yab,xyc->xabc", state_mps[site], ZZ_left)
        state_mps[site] = tmp.reshape(tmp.shape[0], tmp.shape[1], tmp.shape[2] * tmp.shape[3])
    for i in range(1, L // 2, 2):
        site = 2 * i
        tmp = np.einsum("yab,xyc->xacb", state_mps[site], ZZ_right)
        state_mps[site] = tmp.reshape(tmp.shape[0], tmp.shape[1] * tmp.shape[2], tmp.shape[3])    
    for i in range(0, L // 2 - 1, 2):
        site = 2 * i + 1
        tmp = np.einsum("xab,cd->xacbd", state_mps[site], ZZ_mid)
        state_mps[site] = tmp.reshape(tmp.shape[0], tmp.shape[1] * tmp.shape[2], tmp.shape[3] * tmp.shape[4])  

    for i in range(1, L // 2 - 1, 2):
        site = 2 * i
        tmp = np.einsum("yab,xyc->xabc", state_mps[site], ZZ_left)
        state_mps[site] = tmp.reshape(tmp.shape[0], tmp.shape[1], tmp.shape[2] * tmp.shape[3])
    for i in range(2, L // 2, 2):
        site = 2 * i
        tmp = np.einsum("yab,xyc->xacb", state_mps[site], ZZ_right)
        state_mps[site] = tmp.reshape(tmp.shape[0], tmp.shape[1] * tmp.shape[2], tmp.shape[3])    
    for i in range(1, L // 2 - 1, 2):
        site = 2 * i + 1
        tmp = np.einsum("xab,cd->xacbd", state_mps[site], ZZ_mid)
        state_mps[site] = tmp.reshape(tmp.shape[0], tmp.shape[1] * tmp.shape[2], tmp.shape[3] * tmp.shape[4])  

    # odd site ZZ
    for i in range(0, L // 2 - 1, 2):
        site = 2 * i + 1
        tmp = np.einsum("yab,xyc->xabc", state_mps[site], ZZ_left)
        state_mps[site] = tmp.reshape(tmp.shape[0], tmp.shape[1], tmp.shape[2] * tmp.shape[3])
    for i in range(1, L // 2, 2):
        site = 2 * i + 1
        tmp = np.einsum("yab,xyc->xacb", state_mps[site], ZZ_right)
        state_mps[site] = tmp.reshape(tmp.shape[0], tmp.shape[1] * tmp.shape[2], tmp.shape[3])    
    for i in range(0, L // 2 - 1, 2):
        site = 2 * i + 2
        tmp = np.einsum("xab,cd->xacbd", state_mps[site], ZZ_mid)
        state_mps[site] = tmp.reshape(tmp.shape[0], tmp.shape[1] * tmp.shape[2], tmp.shape[3] * tmp.shape[4])  

    for i in range(1, L // 2 - 1, 2):
        site = 2 * i + 1
        tmp = np.einsum("yab,xyc->xabc", state_mps[site], ZZ_left)
        state_mps[site] = tmp.reshape(tmp.shape[0], tmp.shape[1], tmp.shape[2] * tmp.shape[3])
    for i in range(2, L // 2, 2):
        site = 2 * i + 1
        tmp = np.einsum("yab,xyc->xacb", state_mps[site], ZZ_right)
        state_mps[site] = tmp.reshape(tmp.shape[0], tmp.shape[1] * tmp.shape[2], tmp.shape[3])    
    for i in range(1, L // 2 - 1, 2):
        site = 2 * i + 2
        tmp = np.einsum("xab,cd->xacbd", state_mps[site], ZZ_mid)
        state_mps[site] = tmp.reshape(tmp.shape[0], tmp.shape[1] * tmp.shape[2], tmp.shape[3] * tmp.shape[4])  
    
    # for site in range(L):
    #     print("site", site, "shape", state_mps[site].shape)

    zxz_state = MPS_basic_spin_one_half(state_mps)
    p=0.5
    quantum_channel = np.kron(id, id) * (1-p) + np.kron(X, X) * p
    quantum_channel = quantum_channel.reshape(2, 2, 2, 2)
    corr = zxz_state.renyi_2_correlator(L // 8 * 2 + 1, L - L // 8 * 2 - 1, Z, quantum_channel, [2*i for i in range(L//2)])
    #print("corr", corr)

    return corr

#print(build_zxz_state_and_compute_corr(L=20, x=0.12))