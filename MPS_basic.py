import numpy as np
import torch


class MPS_basic:
    def __init__(self, state, phys_first: bool = True):
        """
        state: list of site tensors. This class contracts tensors assuming the
        physical leg is first: shape (d, Dl, Dr).
        If your tensors are instead (Dl, d, Dr), set phys_first=False to
        transpose them into the expected ordering.
        """
        if phys_first:
            self.state = state
        else:
            self.state = [s.transpose(1, 0, 2) for s in state]
        self.L = len(self.state)

    def norm(self):
        operator_right = np.eye(1)
        for i in range(self.L-1, -1, -1):
            operator_right = np.einsum('cf,bac->abf', operator_right, self.state[i])
            operator_right = np.einsum('aef,edf->ad', operator_right, self.state[i].conjugate())
        return operator_right[0, 0]

    def measure_site(self, site, operator):
        operator_right = np.eye(self.state[site].shape[2])
        for i in range(site, -1, -1):
            operator_right = np.einsum('cf,bac->abf', operator_right, self.state[i])
            if i == site:
                operator_right = np.einsum('abf,be->aef', operator_right, operator)
            operator_right = np.einsum('aef,edf->ad', operator_right, self.state[i].conjugate())
        return operator_right[0, 0]

    def measure_site_list(self, site_list, operator_list):
        operator_right = np.eye(self.state[self.L-1].shape[2])
        for i in range(self.L - 1, -1, -1):
            operator_right = np.einsum('cf,bac->abf', operator_right, self.state[i])
            if i in site_list:
                operator_right = np.einsum('abf,be->aef', operator_right, operator_list[site_list.index(i)])
            operator_right = np.einsum('aef,edf->ad', operator_right, self.state[i].conjugate())
        return operator_right[0, 0]
    
    def renyi_2_correlator(self, site_1, site_2, operator, quantum_channel, quantum_channel_list):
        L = len(self.state)
        operator_right = np.ones([1, 1, 1, 1])
        operator_right_0 = np.ones([1, 1, 1, 1])
        for site in range(L-1, -1, -1):
            operator_right = np.einsum("abcd,xgc->abxgd", operator_right, self.state[site])
            operator_right = np.einsum("abxgd,yhd->abxghy", operator_right, self.state[site].conjugate())
            if site in quantum_channel_list:
                operator_right = np.einsum("abwghz,wzxy->abxghy", operator_right, quantum_channel)
            if site == site_1:
                operator_right = np.einsum("abwghz,xw,yz ->abxghy", operator_right, operator, operator.conjugate())
            if site == site_2:
                operator_right = np.einsum("abwghz,xw,yz ->abxghy", operator_right, operator.conjugate().transpose(), operator.transpose())
            if site in quantum_channel_list:
                operator_right = np.einsum("abwghz,xywz->abxghy", operator_right, quantum_channel.conjugate())
            operator_right = np.einsum("abxghy,yea->ebxgh",operator_right, self.state[site])
            operator_right = np.einsum("ebxgh,xfb->efgh", operator_right, self.state[site].conjugate())

            operator_right_0 = np.einsum("abcd,xgc->abxgd", operator_right_0, self.state[site])
            operator_right_0 = np.einsum("abxgd,yhd->abxghy", operator_right_0, self.state[site].conjugate())
            if site in quantum_channel_list:
                operator_right_0 = np.einsum("abwghz,wzxy->abxghy", operator_right_0, quantum_channel)
                operator_right_0 = np.einsum("abwghz,xywz->abxghy", operator_right_0, quantum_channel.conjugate())
            operator_right_0 = np.einsum("abxghy,yea->ebxgh",operator_right_0, self.state[site])
            operator_right_0 = np.einsum("ebxgh,xfb->efgh", operator_right_0, self.state[site].conjugate())

            norm = np.linalg.norm(operator_right_0)
            operator_right, operator_right_0 = operator_right / norm, operator_right_0 / norm

        corr = operator_right[0, 0, 0, 0] / operator_right_0[0, 0, 0, 0]
        return corr
    
    # def renyi_2_correlator(self, site_1, site_2, operator, quantum_channel, quantum_channel_list, dtype=None):
    #     if dtype is None:
    #         torch_dtype = torch.float32
    #     elif isinstance(dtype, torch.dtype):
    #         if dtype.is_complex:
    #             torch_dtype = torch.complex64
    #         else:
    #             torch_dtype = torch.float32
    #     else:
    #         try:
    #             np_dtype = np.dtype(dtype)
    #         except TypeError:
    #             torch_dtype = torch.float32
    #         else:
    #             if np_dtype.kind == "c":
    #                 torch_dtype = torch.complex64
    #             else:
    #                 torch_dtype = torch.float32
    #     return renyi_2_correlator_torch(
    #         self.state,
    #         site_1,
    #         site_2,
    #         operator,
    #         quantum_channel,
    #         quantum_channel_list,
    #         dtype=torch_dtype,
    #     )
    
    # def renyi_2_correlator_mpo(self, site_1, site_2, operator, quantum_channel_mpo):
    #     # abcd,efgh  boundary indices
    #     # xy,zw physical indices
    #     # mn,op virtual indices of quantum channel

    #     L = len(self.state)
    #     operator_right = np.ones([1, 1, 1, 1, 1, 1])
    #     operator_right_0 = np.ones([1, 1, 1, 1, 1, 1])
    #     for site in range(L-1, -1, -1):
    #         quantum_channel = quantum_channel_mpo[site]
    #         operator_right = np.einsum("abcdmn,xgc->abxgdmn", operator_right, self.state[site])
    #         operator_right = np.einsum("abxgdmn,yhd->abxghymn", operator_right, self.state[site].conjugate())
            
    #         operator_right = np.einsum("abwghzmn,wzxypn->abxghymp", operator_right, quantum_channel)

    #         if site == site_1:
    #             operator_right = np.einsum("abwghzmp,xw,yz ->abxghymp", operator_right, operator, operator.conjugate())
    #         if site == site_2:
    #             operator_right = np.einsum("abwghzmp,xw,yz ->abxghymp", operator_right, operator.conjugate().transpose(), operator.transpose())
            
    #         operator_right = np.einsum("abwghzmp,xywzom->abxghyop", operator_right, quantum_channel.conjugate())
            
    #         operator_right = np.einsum("abxghyop,yea->ebxghop",operator_right, self.state[site])
    #         operator_right = np.einsum("ebxghop,xfb->efghop", operator_right, self.state[site].conjugate())

    #         operator_right_0 = np.einsum("abcdmn,xgc->abxgdmn", operator_right_0, self.state[site])
    #         operator_right_0 = np.einsum("abxgdmn,yhd->abxghymn", operator_right_0, self.state[site].conjugate())

    #         operator_right_0 = np.einsum("abwghzmn,wzxypn->abxghymp", operator_right_0, quantum_channel)
    #         operator_right_0 = np.einsum("abwghzmp,xywzom->abxghyop", operator_right_0, quantum_channel.conjugate())

    #         operator_right_0 = np.einsum("abxghyop,yea->ebxghop",operator_right_0, self.state[site])
    #         operator_right_0 = np.einsum("ebxghop,xfb->efghop", operator_right_0, self.state[site].conjugate())

    #         norm = np.linalg.norm(operator_right_0)
    #         operator_right, operator_right_0 = operator_right / norm, operator_right_0 / norm

    #     corr = operator_right[0, 0, 0, 0, 0, 0] / operator_right_0[0, 0, 0, 0, 0, 0]
    #     return corr
    
    def renyi_2_correlator_mpo(self, site_1, site_2, operator, quantum_channel_mpo, dtype=None):
        if dtype is None:
            torch_dtype = torch.float32
        elif isinstance(dtype, torch.dtype):
            if dtype.is_complex:
                torch_dtype = torch.complex64
            else:
                torch_dtype = torch.float32
        else:
            try:
                np_dtype = np.dtype(dtype)
            except TypeError:
                torch_dtype = torch.float32
            else:
                if np_dtype.kind == "c":
                    torch_dtype = torch.complex64
                else:
                    torch_dtype = torch.float32
        return renyi_2_correlator_mpo_torch(
            self.state,
            site_1,
            site_2,
            operator,
            quantum_channel_mpo,
            dtype=torch_dtype,
        )

    def check_renyi_2_correlator(self):
        id = np.array([[1, 0], [0, 1]])
        sx = np.array([[0, 1], [1, 0]])
        sz = np.array([[1, 0], [0, -1]])
        s_plus = np.array([[0, 1], [0, 0]])
        s_minus = np.array([[0, 0], [1, 0]])

        p = 0.3
        quantum_channel = (1 - p) * np.kron(id, id) + p * np.kron(sz, sz)
        for site in range(self.L):
            n1, n2, n3 = self.state[site].shape
            self.state[site] = np.tensordot(self.state[site], self.state[site].conj(), axes=0).transpose(0, 3, 1, 4, 2, 5).reshape(n1 ** 2, n2 ** 2, n3 ** 2)
        for site in range(self.L):
            self.state[site] = np.einsum("abc,da->dbc", self.state[site], quantum_channel)

        corr = self.measure_site_list([self.L // 4, 3 * self.L // 4], [np.kron(s_plus, s_plus), np.kron(s_minus, s_minus)])

        norm = self.norm()
        return corr / norm
    
    def check_renyi_2_correlator_mpo(self):
        id = np.array([[1, 0], [0, 1]])
        sx = np.array([[0, 1], [1, 0]])
        sz = np.array([[1, 0], [0, -1]])
        s_plus = np.array([[0, 1], [0, 0]])
        s_minus = np.array([[0, 0], [1, 0]])

        p = 0.3
        quantum_channel_left = np.zeros([4, 4, 2])
        quantum_channel_left[:, :, 0] = np.sqrt(1-p) * np.kron(id, id)
        quantum_channel_left[:, :, 1] = np.sqrt(p) * np.kron(sz, sz)
        quantum_channel_right = np.zeros([4, 4, 2])
        quantum_channel_right[:, :, 0] = np.sqrt(1-p) * np.kron(id, id)
        quantum_channel_right[:, :, 1] = np.sqrt(p) * np.kron(sz, sz)
        for site in range(self.L):
            n1, n2, n3 = self.state[site].shape
            self.state[site] = np.tensordot(self.state[site], self.state[site].conj(), axes=0).transpose(0, 3, 1, 4, 2, 5).reshape(n1 ** 2, n2 ** 2, n3 ** 2)
        for site in range(self.L-1):
            self.state[site] = np.einsum("abc,dax->dbxc", self.state[site], quantum_channel_left)
            D0, D1, D2, D3 = self.state[site].shape
            self.state[site] = self.state[site].reshape(D0, D1, D2*D3)

            self.state[site + 1] = np.einsum("abc,day->dybc", self.state[site + 1], quantum_channel_right)
            D0, D1, D2, D3 = self.state[site + 1].shape
            self.state[site + 1] = self.state[site + 1].reshape(D0, D1*D2, D3)

        corr = self.measure_site_list([self.L // 4, 3 * self.L // 4], [np.kron(s_plus, s_plus), np.kron(s_minus, s_minus)])
        # corr = self.measure_site_list([self.L // 4, 3 * self.L // 4], [np.kron(sz, sz), np.kron(sz, sz)])
        norm = self.norm()
        return corr / norm

    def mixed_correlator(self, site_1, site_2, operator, quantum_channel, quantum_channel_list):
        L = len(self.state)
        operator_right = np.ones([1, 1])
        operator_right_0 = np.ones([1, 1])
        for site in range(L-1, -1, -1):
            operator_right = np.einsum("cd,xgc->xgd", operator_right, self.state[site])
            operator_right = np.einsum("xgd,yhd->xghy", operator_right, self.state[site].conjugate())
            if site in quantum_channel_list:
                operator_right = np.einsum("wghz,wzxy->xghy", operator_right, quantum_channel)
            if site == site_1:
                operator_right = np.einsum("wghy,xw->xghy", operator_right, operator)
            if site == site_2:
                operator_right = np.einsum("wghy,xw->xghy", operator_right, operator.conjugate().transpose())
            operator_right = np.einsum("xghx->gh",operator_right)

            operator_right_0 = np.einsum("cd,xgc->xgd", operator_right_0, self.state[site])
            operator_right_0 = np.einsum("xgd,yhd->xghy", operator_right_0, self.state[site].conjugate())
            if site in quantum_channel_list:
                operator_right_0 = np.einsum("wghz,wzxy->xghy", operator_right_0, quantum_channel)
            operator_right_0 = np.einsum("xghx->gh",operator_right_0)

            norm = np.linalg.norm(operator_right_0)
            operator_right, operator_right_0 = operator_right / norm, operator_right_0 / norm

        corr = operator_right[0, 0] / operator_right_0[0, 0]
        return corr

class MPS_basic_spin_one_half(MPS_basic):
    def __init__(self, state, phys_first: bool = True):
        super().__init__(state, phys_first=phys_first)
        self.id = np.array([[1, 0], [0, 1]])
        self.sx = np.array([[0, 1], [1, 0]])
        self.sz = np.array([[1, 0], [0, -1]])
        self.isy = np.array([[0, 1], [-1, 0]])
        self.s_plus = np.array([[0, 1], [0, 0]])
        self.s_minus = np.array([[0, 0], [1, 0]])





def renyi_2_correlator_torch(
    state,
    site_1,
    site_2,
    operator,
    quantum_channel,
    quantum_channel_list,
    device=None,
    dtype=torch.float32,
):
    # --- select device ---
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- move inputs to GPU / correct dtype ---
    torch_state = []
    for s in state:
        if isinstance(s, torch.Tensor):
            torch_state.append(s.to(device=device, dtype=dtype))
        else:
            torch_state.append(torch.as_tensor(s, device=device, dtype=dtype))
    state = torch_state

    operator = torch.as_tensor(operator, device=device, dtype=dtype)
    quantum_channel = torch.as_tensor(quantum_channel, device=device, dtype=dtype)
    operator_conj = operator.conj()
    operator_t = operator.transpose(-2, -1)
    operator_conj_t = operator_conj.transpose(-2, -1)
    quantum_channel_conj = quantum_channel.conj()

    # initialize environment tensors
    operator_right  = torch.ones((1, 1, 1, 1), device=device, dtype=dtype)
    operator_right_0 = torch.ones((1, 1, 1, 1), device=device, dtype=dtype)

    L = len(state)

    # loop from right to left
    for site in range(L - 1, -1, -1):
        psi = state[site]
        psi_conj = psi.conj()

        # --- operator_right branch ---
        operator_right = torch.einsum("abcd,xgc->abxgd", operator_right, psi)
        operator_right = torch.einsum("abxgd,yhd->abxghy", operator_right, psi_conj)

        if site in quantum_channel_list:
            operator_right = torch.einsum("abwghz,wzxy->abxghy", operator_right, quantum_channel)

        if site == site_1 :
            operator_right = torch.einsum("abwghz,xw,yz->abxghy", operator_right, operator, operator_conj)
        if site == site_2:
            operator_right = torch.einsum("abwghz,xw,yz->abxghy", operator_right, operator_conj_t, operator_t)

        if site in quantum_channel_list:
            operator_right = torch.einsum("abwghz,xywz->abxghy",
                                          operator_right, quantum_channel_conj)

        operator_right = torch.einsum("abxghy,yea->ebxgh", operator_right, psi)
        operator_right = torch.einsum("ebxgh,xfb->efgh", operator_right, psi_conj)

        # --- operator_right_0 branch (no operator insertion) ---
        operator_right_0 = torch.einsum("abcd,xgc->abxgd", operator_right_0, psi)
        operator_right_0 = torch.einsum("abxgd,yhd->abxghy", operator_right_0, psi_conj)

        if site in quantum_channel_list:
            operator_right_0 = torch.einsum("abwghz,wzxy->abxghy",
                                            operator_right_0, quantum_channel)
            operator_right_0 = torch.einsum("abwghz,xywz->abxghy",
                                            operator_right_0, quantum_channel_conj)

        operator_right_0 = torch.einsum("abxghy,yea->ebxgh", operator_right_0, psi)
        operator_right_0 = torch.einsum("ebxgh,xfb->efgh", operator_right_0, psi_conj)

        # normalization
        norm = torch.linalg.norm(operator_right_0)
        operator_right  = operator_right / norm
        operator_right_0 = operator_right_0 / norm

    # final correlator (scalar)
    corr = operator_right[0, 0, 0, 0] / operator_right_0[0, 0, 0, 0]
    return corr.item()


def renyi_2_correlator_mpo_torch(
    state,
    site_1,
    site_2,
    operator,
    quantum_channel_mpo,
    device=None,
    dtype=torch.float32,
):
    """GPU version of renyi_2_correlator_mpo; follows the numpy implementation."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch_state = []
    for s in state:
        if isinstance(s, torch.Tensor):
            torch_state.append(s.to(device=device, dtype=dtype))
        else:
            torch_state.append(torch.as_tensor(s, device=device, dtype=dtype))
    state = torch_state

    operator = torch.as_tensor(operator, device=device, dtype=dtype)
    operator_conj = operator.conj()
    operator_t = operator.transpose(-2, -1)
    operator_conj_t = operator_conj.transpose(-2, -1)
    quantum_channel_mpo = [
        torch.as_tensor(qc, device=device, dtype=dtype)
        for qc in quantum_channel_mpo
    ]

    L = len(state)
    operator_right = torch.ones((1, 1, 1, 1, 1, 1), device=device, dtype=dtype)
    operator_right_0 = torch.ones((1, 1, 1, 1, 1, 1), device=device, dtype=dtype)

    for site in range(L - 1, -1, -1):
        psi = state[site]
        psi_conj = psi.conj()
        quantum_channel = quantum_channel_mpo[site]
        quantum_channel_conj = quantum_channel.conj()

        operator_right = torch.einsum("abcdmn,xgc->abxgdmn", operator_right, psi)
        operator_right = torch.einsum("abxgdmn,yhd->abxghymn", operator_right, psi_conj)
        operator_right = torch.einsum("abwghzmn,wzxypn->abxghymp", operator_right, quantum_channel)

        if site == site_1:
            operator_right = torch.einsum(
                "abwghzmp,xw,yz->abxghymp",
                operator_right,
                operator,
                operator_conj,
            )
        if site == site_2:
            operator_right = torch.einsum(
                "abwghzmp,xw,yz->abxghymp",
                operator_right,
                operator_conj_t,
                operator_t,
            )

        operator_right = torch.einsum(
            "abwghzmp,xywzom->abxghyop", operator_right, quantum_channel_conj
        )
        operator_right = torch.einsum("abxghyop,yea->ebxghop", operator_right, psi)
        operator_right = torch.einsum("ebxghop,xfb->efghop", operator_right, psi_conj)

        operator_right_0 = torch.einsum("abcdmn,xgc->abxgdmn", operator_right_0, psi)
        operator_right_0 = torch.einsum("abxgdmn,yhd->abxghymn", operator_right_0, psi_conj)
        operator_right_0 = torch.einsum("abwghzmn,wzxypn->abxghymp", operator_right_0, quantum_channel)
        operator_right_0 = torch.einsum(
            "abwghzmp,xywzom->abxghyop", operator_right_0, quantum_channel_conj
        )
        operator_right_0 = torch.einsum("abxghyop,yea->ebxghop", operator_right_0, psi)
        operator_right_0 = torch.einsum("ebxghop,xfb->efghop", operator_right_0, psi_conj)

        norm = torch.linalg.norm(operator_right_0)
        operator_right = operator_right / norm
        operator_right_0 = operator_right_0 / norm

    corr = operator_right[0, 0, 0, 0, 0, 0] / operator_right_0[0, 0, 0, 0, 0, 0]
    return corr.item()
