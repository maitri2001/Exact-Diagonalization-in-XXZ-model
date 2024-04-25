#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""


@author: maitri
"""

# ----------Exact Diagonalization Of XXZ model and numerical study of Thermalization in this model-----------
# -----------FOR THIS EXAMPLE CODE I AM TAKING L = 8 AS THE SYSTEM SIZE-------------


import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting library
from scipy.sparse import csr_matrix # for sparse matrix tranformation
from scipy.sparse import csr_matrix # for sparse matrix tranformation
import cmath
from collections import Counter
from scipy.stats import gaussian_kde


#----------------- Parameters of the system----------------------------------------
Delta = 0.5
J = cmath.sqrt(-1)
L = 8
lamb = 1



# to convert decimal to binary with L number of digits(sites)


def decToBin(dec, L):
    binaryString = bin(dec)[2:] 
    paddedBinaryString = binaryString.rjust(L, '0')  
    Bin = [int(bit) for bit in paddedBinaryString]
    return Bin


def binToDec(Bin):
    binary_string = ''.join(map(str, Bin))
    dec = int(binary_string, 2)
    return dec


def bit_flip(state, position1, position2):  # state is in binary representation
    # Perform a bit flip between position1 and position2 in the state
    new_state = state.copy()
    new_state[position1], new_state[position2] = new_state[position2], new_state[position1]
    return new_state


# ------------------ Creating the action of the Hamiltonian--------------------


def Act_H(dec): # Hamiltonian acting on the decimal representation of a basis state |n>
    n = decToBin(dec, L)
    output = []
    summ = 0

    for i in np.arange(0, L):
        j1 = (i+1) % L
        j2 = (i+2) % L
        summ = summ + (-Delta*0.5/(1+lamb))*(2*n[i]-1)*(
            2*n[j1]-1) + lamb*(-Delta*0.5/(1+lamb))*(2*n[i]-1)*(2*n[j2]-1)
    # Appending the density - density interaction contribution
    output.append([n, summ])

    m = n.copy()  # incorporating nearest and next nearest neighbor hopping contributions
    for i in np.arange(0, L):
        j1 = (i+1) % L
        j2 = (i+2) % L
        if n[i] != n[j1]:
            m = bit_flip(n, i, j1)
            output.append([m, -1/(1+lamb)])
        m = n.copy()
        if n[i] != n[j2]:
            m = bit_flip(n, i, j2)
            output.append([m, -1*lamb/(1+lamb)])
    return output  # gives in Fock state form with corresponding weights

# ------------------------------------- Creating the Hamiltonian in full basis--------------


def H_full_basis(L):
    H = np.zeros((2**L, 2**L))
    for m in range(2**L):
        out_put = Act_H(m)
        for listi in out_put:
            strn = listi[0]
            n = binToDec(strn)
            weight = listi[1]
            H[n][m] += weight
    return H

eg_val_full_Hilbert_space = np.linalg.eigh( H_full_basis(L))[0] # Diagonalizing the full Hamiltonian directly

# ------------------------------------------------------------------------------------------


# state consists of two-element lists , 1st is string representation and 2nd is the corresponding weight
# The following function converts this state into a column with 2**L elements, where the number in the i'th position is the weight of the basis state 
# with integer representation 'i'

def colmn_full_basis(state):
    psi = np.zeros(2**L, dtype=np.complex128)
    for listi in state:
        str_dec = binToDec(listi[0])
        weight = listi[1]
        psi[str_dec] += weight
    return psi

# ------------------------Creating the action of the A operator(eq. 28) in the paper)--------------


def Act_A(dec):
    n = decToBin(dec, L)
    output = []
    summ = 0
    for l in range(L):
        for m in range(L):
            if n[l] == 0 and n[m] == 1:
                p = bit_flip(n, l, m)
                output.append([p, 1/L])
        summ = summ + (1/L)*n[l]

    output.append([n, summ])
    return output  # gives in Fock state form with corr weights

# ------------------------- Creating the A matrix in full basis ----------------------------


def A_full_basis(L):
    A = np.zeros((2**L, 2**L))
    for m in range(2**L):
        out_put = Act_A(m)
        for listi in out_put:
            strn = listi[0]
            n = binToDec(strn)
            weight = listi[1]
            A[n][m] += weight
    return A


# ------------------------Creating the action of the B operator(eq. 29) in the paper)--------------

def Act_B(dec):
    n = decToBin(dec, L)
    output = []
    summ = 0
    for l in range(L):
        j = (l+1) % L
        summ = summ + (1/L) * n[l]*n[j]
    output.append([n, summ])
    return output  # gives in Fock state form with corr weights


# ------------------------- Creating the B matrix in full basis ----------------------------


def B_full_basis(L):
    B = np.zeros((2**L, 2**L))
    for m in range(2**L):
        out_put = Act_B(m)
        for listi in out_put:
            strn = listi[0]
            n = binToDec(strn)
            weight = listi[1]
            B[n][m] += weight
    return B


# -------------------------------------------------------------------------------------------

# Constructing the basis state for SN where SN is the subspace i.e, basis for N particle sector

def Basis(N):
    basisN = []
    for dec in np.arange(0, 2**L, 1):
        n = decToBin(dec, L)
        if sum(n) == N:
            basisN.append(n)
    return basisN  # it will give the fock state rep


def Basis2(N):
    basisN = []
    for dec in np.arange(0, 2**L, 1):
        n = decToBin(dec, L)
        if sum(n) == N:
            basisN.append(binToDec(n))
    return basisN  # it will give integer rep of   the fock state


#----------------------------------------------------------------------------------------------------#

######---------- Creating a block Hamiltonian for a particular filling N-------------------#


def build_HN(L, N):
    HNT = []
    BasisN = Basis(N)
    for i in BasisN:
        j = binToDec(i)
        output = Act_H(j)
        row = []
        for k in BasisN:
            # We are using the if condition because we may get not get the state in ouput with which we are starting to search
            if k in [sublist[0] for sublist in output]:
                ind_of_k = [sublist[0] for sublist in output].index(k)
                weight = output[ind_of_k][1]
                row.append(weight)
            else:  # if k is not in output the correseponding matrix element is zero
                row.append(0)
        HNT.append(row)
    return np.transpose(HNT)


# ------------------------------------Translational operators-------------------------------------------------


# we have  the translation symmetry for this model and we are using the periodic boundary condition
# List of representating states that will contribute to SN,k

def Translation(state, L):  # state is in binary here as input
    new_state = np.zeros(L, dtype=int)
    for i in np.arange(0, L, 1):
        new_state[(i+1) % L] = state[i]
    return new_state.tolist()


# l is the number of times the Translational operator has been acted
def multi_Translation(state, l, L):

    new_state2 = [int(k) for k in state]
    for i in range(l):
        new_state2 = Translation(new_state2, L)
    return new_state2


# ------------------------------------------------------------------------------------------------------
#---------------------- Creating the BasisNK function--------------------------------------------


def BasisNK(L, N, k):  
    basisNk = []  # We are storing here the respresentating state in Fock space rep not in integer rep
    V = []
    d = []
    u_values = {}
    Basis2N = Basis2(N)
    for i in Basis2N:
        if i not in V:
            u_list = []
            u_list.append(i)
            u = i
            # List to store u values for a particular i
            for j in np.arange(1, L+1, 1):
                u = binToDec(Translation(decToBin(u, L), L))
                if u != i:
                    V.append(u)
                    u_list.append(u)
                elif u==i and (j*k)%L == 0:
                    basisNk.append(decToBin(i, L))
                    V.append(i)
                    d.append(j)
                    u_values[i] = u_list
                    break
                else:
                    break
    return (" n_bar=", basisNk, " period=", d, " Connected subspaces=", u_values)




# -------------------------------------------------------------------------------------------------

#creating the Hamiltonian in the subspace S_{N,k}, which consists of states with N particles and Translation operator eigenvalue exp(2*pi*i*k/L)

def HNk(L, N, k):
    HNkT = []
    BasisNKLNk = BasisNK(L, N, k)
    for n_bar in BasisNKLNk[1]:
        period_n_bar = BasisNKLNk[3][BasisNKLNk[1].index(n_bar)]
        i = binToDec(n_bar)
        output = Act_H(i)
        row = []
        for m_bar in BasisNKLNk[1]:
            period_m_bar = BasisNKLNk[3][BasisNKLNk[1].index(m_bar)]
            index_m_bar = BasisNKLNk[1].index(m_bar)
            summ = 0
            # Extracting the desired list from the dictionary
            list_of_connection = BasisNKLNk[5].get(binToDec(m_bar), [])
            for m_dec in list_of_connection:
                m = decToBin(m_dec, L)
                # We are using the if condition because we may get not get the state in ouput with which we are starting to search
                if m in [sublist[0] for sublist in output]:
                    ind_of_m = [sublist[0] for sublist in output].index(m)
                    weight = output[ind_of_m][1]

                else:  # if k is not in output the correseponding matrix element is zero
                    weight = 0
                # Finding the position of the connected state with the Rs state to get the period
                d_m = list_of_connection.index(m_dec)
                summ = summ + weight * np.exp(2*J*(np.pi)*k*(1/L))**d_m
            total_sum = summ*np.sqrt(period_n_bar/period_m_bar)
            row.append(total_sum)
        HNkT.append(row)
    return np.transpose(HNkT)

# --------------------------------------------------------------------------------------------------------------------------


# ------------------ Define the Reflection R, spin-flip operator X acting on basis state--------------------------------

def R(state_i):  # state is in binary form
    state = state_i.copy()
    for i in np.arange(0, int((L)/2), 1):
        state = bit_flip(state, i, (L-1)-i)
    return state


def X(state_i):  # state is in binary form
    state = state_i.copy()
    for i in np.arange(0, L, 1):
        if state[i] == 0:
            state[i] = 1
        else:
            state[i] = 0
    return state



# ------------------------ Build the SRS for the basis state of MSS-------------------------------


def find_RS(state, L): # finding the representative state of a given state
    EC = []
    u = state.copy()
    for i in range(L):
        # Here we may have double counting of the state but we are interesed in min integer of state, so no matter
        EC.append(binToDec(u))
        u = Translation(u, L)
    n = min(EC)
    return decToBin(n, L)

# --------------------building the MSS basis-------------------------------------


basisNk_half_fill = BasisNK(L, L/2, 0) # storing the BasisNK(L, L/2, 0) for convenience to use later


def SRS_basis_MSS(L):
    SRS_list_MSS = []
    SRS_multiplicity = []
    SEC_list = {}
    for n_bar in basisNk_half_fill[1]:  # Maximum symmetry sector
        super_equivalence_class = []
        n_barX = find_RS(X(n_bar), L)
        n_barR = find_RS(R(n_bar), L)
        n_barRX = find_RS(R(X(n_bar)), L)
        dec_rep = [binToDec(n_bar), binToDec(n_barX),
                   binToDec(n_barR), binToDec(n_barRX)]
        dec_rep_set = set(dec_rep)
        if binToDec(n_bar) <= min(dec_rep):
            SRS_list_MSS.append(n_bar)
            SRS_multiplicity.append(len(set(dec_rep)))
            for n in dec_rep_set:
                super_equivalence_class = super_equivalence_class + basisNk_half_fill[5].get(n, [])
            SEC_list[binToDec(n_bar)] = super_equivalence_class

    return ["the SRS states =", SRS_list_MSS, "the multiplicities =", SRS_multiplicity, "SECs =", SEC_list]



# -------------------MSS Basis states in occupation number basis--------------

# Given a general state in MSS basis, the following function calculates the state in occupation
# number basis of the full Hilbert space of the form [[|n>,a_n],.....] which will be useful for
# calculating Entanglement Entropy


SRS_basis_MSS = SRS_basis_MSS(L) # storing the data for convenience to use later on


def rep_MSS_state(state):  # input state is in MSS basis
    state_in_occcupation_basis = []
    srs_states = SRS_basis_MSS[1]
    for i in range(len(srs_states)):
        srs = srs_states[i]
        b = state[i]
        period_srs = basisNk_half_fill[3][basisNk_half_fill[1].index(srs)]
        multi_srs = SRS_basis_MSS[3][SRS_basis_MSS[1].index(srs)]
        z_srs = cmath.sqrt(period_srs * multi_srs)/(4*L)
        weight = b * z_srs
        if weight != 0:
            for l in range(L):
                state1 = multi_Translation(srs, l, L)
                state2 = X(state1)
                state3 = R(state1)
                state4 = X(state3)
                state_in_occcupation_basis.append([state1, weight])
                state_in_occcupation_basis.append([state2, weight])
                state_in_occcupation_basis.append([state3, weight])
                state_in_occcupation_basis.append([state4, weight])

    return state_in_occcupation_basis


# ----------------------------Matrix element of HMSS(Maximal symmetry sector) ----------------------------

##
def HMSS(L):
    HMSST = []
    
    for n_tilde in SRS_basis_MSS[1]:
        period_n_tilde = basisNk_half_fill[3][basisNk_half_fill[1].index(
            n_tilde)]
        multi_n_tilde = SRS_basis_MSS[3][SRS_basis_MSS[1].index(n_tilde)]
        i = binToDec(n_tilde)
        output = Act_H(i)
        row = []
        
        for m_tilde in SRS_basis_MSS[1]:
            period_m_tilde = basisNk_half_fill[3][basisNk_half_fill[1].index(
                m_tilde)]
            multi_m_tilde = SRS_basis_MSS[3][SRS_basis_MSS[1].index(m_tilde)]
            index_m_tilde = basisNk_half_fill[1].index(m_tilde)
            summ = 0
            # Extracting the desired list from the dictionary
            list_of_connection = SRS_basis_MSS[5].get(binToDec(m_tilde), [])
            
            for m_dec in list_of_connection:
                m = decToBin(m_dec, L)
                # We are using the if condition because we may get not get the state in ouput with which we are starting to search
                if m in [sublist[0] for sublist in output]:
                    ind_of_m = [sublist[0] for sublist in output].index(m)
                    weight = output[ind_of_m][1]

                else:  # if k is not in output the correseponding matrix element is zero
                    weight = 0
                
                summ = summ + weight
            total_sum = summ * np.sqrt((period_n_tilde*multi_n_tilde)/(period_m_tilde*multi_m_tilde))
            row.append(total_sum)
        HMSST.append(row)
        
    return np.transpose(HMSST)


# -------------------Build A matrix in MSS basis--------------------------

def AMSS(L):
    AMSST = []
    
    for n_tilde in SRS_basis_MSS[1]:
        period_n_tilde = basisNk_half_fill[3][basisNk_half_fill[1].index(
            n_tilde)]
        multi_n_tilde = SRS_basis_MSS[3][SRS_basis_MSS[1].index(n_tilde)]
        i = binToDec(n_tilde)
        output = Act_A(i)
        row = []
        
        for m_tilde in SRS_basis_MSS[1]:
            period_m_tilde = basisNk_half_fill[3][basisNk_half_fill[1].index(
                m_tilde)]
            multi_m_tilde = SRS_basis_MSS[3][SRS_basis_MSS[1].index(m_tilde)]
            index_m_tilde = basisNk_half_fill[1].index(m_tilde)
            summ = 0
            # Extracting the desired list from the dictionary
            list_of_connection = SRS_basis_MSS[5].get(binToDec(m_tilde), [])
            
            for m_dec in list_of_connection:
                m = decToBin(m_dec, L)
                # We are using the if condition because we may get not get the state in ouput with which we are starting to search
                if m in [sublist[0] for sublist in output]:
                    ind_of_m = [sublist[0] for sublist in output].index(m)
                    weight = output[ind_of_m][1]
                    

                else:  # if k is not in output the correseponding matrix element is zero
                    weight = 0
                
                summ = summ + weight
            total_sum = summ * np.sqrt((period_n_tilde*multi_n_tilde)/(period_m_tilde*multi_m_tilde))
            row.append(total_sum)
        AMSST.append(row)
        
    return np.transpose(AMSST)


# -------------------Build B matrix in MSS basis--------------------------
def BMSS(L):
    BMSST = []
    
    for n_tilde in SRS_basis_MSS[1]:
        period_n_tilde = basisNk_half_fill[3][basisNk_half_fill[1].index(
            n_tilde)]
        multi_n_tilde = SRS_basis_MSS[3][SRS_basis_MSS[1].index(n_tilde)]
        i = binToDec(n_tilde)
        output = Act_B(i)
        row = []
        
        for m_tilde in SRS_basis_MSS[1]:
            period_m_tilde = basisNk_half_fill[3][basisNk_half_fill[1].index(
                m_tilde)]
            multi_m_tilde = SRS_basis_MSS[3][SRS_basis_MSS[1].index(m_tilde)]
            index_m_tilde = basisNk_half_fill[1].index(m_tilde)
            summ = 0
            # Extracting the desired list from the dictionary
            list_of_connection = SRS_basis_MSS[5].get(binToDec(m_tilde), [])
            
            for m_dec in list_of_connection:
                m = decToBin(m_dec, L)
                # We are using the if condition because we may get not get the state in ouput with which we are starting to search
                if m in [sublist[0] for sublist in output]:
                    ind_of_m = [sublist[0] for sublist in output].index(m)
                    weight = output[ind_of_m][1]
                    

                else:  # if k is not in output the correseponding matrix element is zero
                    weight = 0
                
                summ = summ + weight
            total_sum = summ * np.sqrt((period_n_tilde*multi_n_tilde)/(period_m_tilde*multi_m_tilde))
            row.append(total_sum)
        BMSST.append(row)
        
    return np.transpose(BMSST)


# ---------------------------------Working in Maximal Symmetry Sector----------------------------------

Energ, U = np.linalg.eigh(HMSS(L)) # calculating energy eigenvalues and eigenvectors of HMSS

U_dagger = U.conj().T

V = np.transpose(U)  # eigen_vector_matrix for HMSS matrix, V contains the eigenvectors as rows


# ------------------------ Time evolution  using Suzuki -Trotter -----------------------
# epsilon is time step

def act_u(state, L, l, m, epsilon):  # sate is in original occupation number basis, it has 2^L elements
    state_prime = np.zeros(2**L, dtype=np.complex128)
    for n in range(2**L):
        n_bin = decToBin(n, L)
        n_l = n_bin[l]
        n_m = n_bin[m]
        if n_l == n_m:
            state_prime[n] += state[n] * cmath.exp(J*Delta * epsilon)
        else:
            state_prime[n] += state[n] * \
                cmath.exp(-J*Delta * epsilon/2)*np.cos(epsilon)
            m_bin = bit_flip(n_bin, l, m)
            m_dec = binToDec(m_bin)
            state_prime[m_dec] += state[n] * J * \
                cmath.exp(-J*Delta * epsilon/2)*np.sin(epsilon)

    return state_prime

#-----------------------------------------------------------------------------------------------------------

def expH0(state, L, eps):
    state_w = state.copy()
    for l in range(L//2):
        state_w = act_u(state_w, L, 2*l % L, (2*l+1) % L, eps)
    return state_w

#-------------------------------------------------------------------------------------------------------------

def expH1(state, L, eps):
    state_w = state.copy()
    for l in range(L//2):
        state_w = act_u(state_w, L, (2*l+1) % L, (2*l+2) % L, eps)
    return state_w

#---------------------------------------------------------------------------------------------------------------

def expH2(state, L, eps):
    state_w = state.copy()
    for l in range(L//4):
        state_w = act_u(state_w, L, 4*l % L, (4*l+2) % L, eps)
        state_w = act_u(state_w, L, (4*l+1) % L, (4*l+3) % L, eps)

    return state_w

#---------------------------------------------------------------------------------------------------------------------

def expH3(state, L, eps):
    state_w = state.copy()
    for l in range(L//4):
        state_w = act_u(state_w, L, (4*l+2) % L, (4*l+4) % L, eps)
        state_w = act_u(state_w, L, (4*l+3) % L, (4*l+5) % L, eps)

    return state_w

#--------------------------------------------------------------------------------------------------------------------------
# Doing the time evolution of the state psi_0 and calculating the expectation value of A and B at each time step

dim_MSS = len(SRS_basis_MSS[1])

ini_state_MSS = np.zeros(dim_MSS)

ini_state_MSS[-1] = 1  # It is also in MSS basis

ini_state = colmn_full_basis(rep_MSS_state(ini_state_MSS)) # initial state psi_0 in the occupation number basis

state_work = ini_state
eps = 0.2
t = 0

A_expc_ST = []
B_expc_ST = []
t_list_ST = []
A_full_basis = A_full_basis(L)
B_full_basis = B_full_basis(L)


for t_step in range(26):
    p1 = (1/(1+lamb))*eps
    p2 = (lamb/(1+lamb))*eps
    x1 = expH0(state_work, L, p1/2)
    x2 = expH1(x1, L, p1/2)
    x3 = expH2(x2, L, p2/2)
    x4 = expH3(x3, L, p2)
    x5 = expH2(x4, L, p2/2)
    x6 = expH1(x5, L, p1/2)
    state_work = expH0(x6, L, p1/2) # time evolved state 
    
    A_expc = np.dot(np.conjugate(state_work),(np.dot(A_full_basis, state_work)))
    A_expc_ST.append(A_expc)
    B_expc = np.dot(np.conjugate(state_work),(np.dot(B_full_basis, state_work)))
    B_expc_ST.append(B_expc)
    
    t += eps
    t_list_ST.append(t)

stacked_ST_t_list_A_B_expc = np.column_stack((t_list_ST, A_expc_ST, B_expc_ST))


#------------------------- Exact time evolution of the operators A and B  for the initial state |psi0> = (1/sqrt(2)(|0101....> + |1010....>))-------------- 

#------------------Time Evolution of A and B in MSS state--------------------------------

dim_MSS = len(SRS_basis_MSS[1])

ini_state = np.zeros(dim_MSS)

ini_state[-1] = 1 # It is also in MSS basis

t_list = np.linspace(0,50,1000)


###----------------function for exact time-evolution of a general state in the MSS----------------------

def psi_t(t,state_MSS):
    ## here state_MSS is in basis of MSS 
    ##  the state |psi(t)> will be returned in MSS basis |n_tilde>_MSS 
    state = np.dot(U_dagger,state_MSS) # This in MSS eigenstate basis
    psi = []
    for i in range(dim_MSS):
        summ = 0
        for alpha in range(dim_MSS):
            summ += state[alpha]*V[alpha][i]*cmath.exp(-J*Energ[alpha]*t)
        psi.append(summ)        
    return np.array(psi)        
 
###-----------------------------------------------------------------------------------------------------



###-----------------------exact time-evolution of expectation values of operators A and B -------------------------------
           
expec_A_t =[] 
expec_B_t =[]  
A = AMSS(L)
B = BMSS(L)
for t in t_list:
    psit = psi_t(t,ini_state)
    A_psi_t = np.dot(A,psit )
    ex_A = np.dot(np.conjugate(psit),A_psi_t)
    B_psi_t = np.dot(B,psit )
    ex_B = np.dot(np.conjugate(psit),B_psi_t)
    expec_A_t.append( ex_A)
    expec_B_t.append( ex_B)
    


data_t_At_Bt = np.column_stack((t_list ,expec_A_t,expec_B_t))

   

#----------------------------Entanglement Entropy-------------------------------------------------------------------------------------------

def Ent(state,L, L1, L2):## state is given in occupation basis of the form [[state,weight],.....]
    Psi = np.zeros((2**L2,2**L1),dtype=np.complex128)
    
    for listi in state:
        a_n = listi[1]
        n = binToDec(listi[0])
        r = n%2**L1
        l = int(n/2**L1)
        Psi[l][r] += a_n      # updating the matrix elemnt of Psi
    U, Sing_val, VT = np.linalg.svd(Psi)
    Entropy = 0
    
    for i in Sing_val:
        if i != 0:
            Entropy += - i**2 * np.log(i**2) 
    return Entropy        
            
#--------------------- Inverse temperature vs average energy---------------------------

def E_av(beta, E_list):
    sum1 = 0 # for the partition function part
    for E in E_list:
        sum1 += np.exp(-beta * E)
        
    sum2= 0
    for E in E_list:
        sum2 += E * np.exp(-beta * E)*(1/sum1)
    return  sum2    

beta_list = np.linspace(0,3,30)  

E8_byL_av_list = [(1/8)*E_av(beta,Energ) for beta in beta_list]  

plt.figure()
plt.plot(E8_byL_av_list ,beta_list,label ="$L = 8$",linestyle ="-.")
plt.legend()
plt.xlabel("E/L")
plt.ylabel(r"$\beta$")
plt.title(r"$\beta$ vs E/L")
plt.show()  


#-------------------------- PLOTs----------------------------

#------------ Eigenvalues vs Quantum number-------------------------
 
eigenvaluesHN = [] 
for N in np.arange(0,L+1 ):
    egvalsHN = (np.linalg.eigvals(build_HN(L,N)).real).tolist() 
    eigenvaluesHN = eigenvaluesHN + egvalsHN

sorted_eigenvaluesHN= sorted(eigenvaluesHN)


x1 = [ (i+1)/(len(sorted_eigenvaluesHN)) for i in range(len(sorted_eigenvaluesHN))]
x1_prime = [ (i+1)/(len((eg_val_full_Hilbert_space))) for i in range(len((eg_val_full_Hilbert_space)))]

plt.figure()
plt.scatter(x1,sorted_eigenvaluesHN,marker='o',label=" Diagonalizing hamiltonian block-wise")
plt.scatter(x1_prime,sorted(eg_val_full_Hilbert_space),marker='o',label="Diagonalizing directly hamiltonian matrix")
plt.legend()
plt.xlabel("normalized quantum number")
plt.ylabel('Eigenvalues')
plt.title("Comparison of the diagonalization of the Hamiltonian directly and block-wise")
plt.show() 

#--------------------- Energy eigenvalues in different sectors-------------------
plt.figure()
x1 = [ (i+1)/(len(sorted_eigenvaluesHN)) for i in range(len(sorted_eigenvaluesHN))] 
plt.scatter(x1,sorted_eigenvaluesHN,marker='o',label="S")  
    
eigenvaluesHNLby2 = (np.linalg.eigvals(build_HN(L,L/2)).real).tolist()
sorted_eigenvaluesHNLby2 = sorted(eigenvaluesHNLby2 )
x2 = [( i+1)/(len(sorted_eigenvaluesHNLby2)) for i in range(len(sorted_eigenvaluesHNLby2))]    
plt.scatter(x2,sorted_eigenvaluesHNLby2,marker='v',label="S_N=L/2")

     
    
    
egvalsHNk = ((np.linalg.eigvals(HNk(L,L/2,0))).real).tolist()
sorted_eigenvaluesHNk = sorted(egvalsHNk)
x3 = [( i+1)/(len(sorted_eigenvaluesHNk)) for i in range(len(sorted_eigenvaluesHNk))]    
plt.scatter(x3,sorted_eigenvaluesHNk,marker='+',label ="S_N=L/2,k=0")

egvalsHMSS = ((np.linalg.eigvals(HMSS(L))).real).tolist()
sorted_eigenvaluesHMSS = sorted(egvalsHMSS ) 
x4 = [ (i+1)/(len(sorted_eigenvaluesHMSS)) for i in range(len(sorted_eigenvaluesHMSS))]   
plt.scatter(x4,sorted_eigenvaluesHMSS,marker='D',label="S_N=L/2,0,+1,+1")

plt.xlabel("Normalized Quantum Number")
plt.ylabel("Eigenvalues")
plt.legend()
plt.title("Energy eigenvalues in different sectors")
plt.show() 



#--------------------- expectation values of operator A & B in eigenstate of MSS against corresponding energy eigenvalues-------------  
    
A = AMSS(L)
B = BMSS(L)
A_list=[]
B_list=[]
for v in V:
    m1 = np.dot(A,v)
    n1 = np.dot(np.conjugate(v),m1)
    m2 = np.dot(B,v)
    n2 = np.dot(np.conjugate(v),m2)
    A_list.append(n1)
    B_list.append(n2)

plt.figure()
plt.scatter(Energ/L,A_list)
plt.title("Expectation Value of A in the MSS")
plt.xlabel("E/L")
plt.show()


plt.figure()
plt.scatter(Energ/L,B_list)
plt.title("Expectation Value of B in the MSS")
plt.xlabel("E/L")
plt.show() 

data_energ_by_L_A_B = np.column_stack((Energ/L,A_list,B_list))

###-----------------plotting and comparing exact and Suzuki-Trotter time evolution of A and B operator------------------------------
# Time evolution data of A and B using Suzuki-Trotter and exact was calculated earlier

plt.figure()
plt.plot(t_list_ST, A_expc_ST,label='Suzuki Trotter A',marker = '^')
plt.plot(t_list, expec_A_t,label='exact A',linestyle='-')
plt.plot(t_list_ST, B_expc_ST,label='Suzuki Trotter B',marker ='v')
plt.plot(t_list, expec_B_t,label='exact B',linestyle='--')
plt.xlabel('time')
plt.ylabel('Expectation value')
plt.legend()
plt.xlim(0,5)
plt.ylim(-0.2,0.6)
plt.xlabel("time")
plt.ylabel("Expectation value")
plt.title(f"$L = {L} ,\epsilon = 0. 2$")
plt.show()

   
#--------------------- Entanglement entropy plots --------------------------------

#-------------------- Calculate the entanglement entropy for the eigenstate in the MSS sector-----------------------------

Ent_entropies_for_all_Eigenstate = []

for i in range (len(V)):
    st = V[i]
    state_in_occu = rep_MSS_state(st)
    ent = Ent(state_in_occu,L,L//2,L//2)
    Ent_entropies_for_all_Eigenstate.append(ent)
    
plt.figure()
plt.scatter(Energ/L,Ent_entropies_for_all_Eigenstate)
plt.xlabel('E/L')
plt.ylabel('Entanglement entropy')
plt.title('Entanglement entropy vs E/L')
plt.show()
data_E_by_L_Ent_entropy = np.column_stack((Energ/L ,Ent_entropies_for_all_Eigenstate))
    

# -----------------------Entanglement entropy vs Subsytem size in particular MSS eigenstate------------------------------

Ent_entropies_for_particular_Eigenstate_1=[]
Ent_entropies_for_particular_Eigenstate_2=[]
listi1 =np.abs(Energ/L).tolist() 
listi2 =np.abs(- 0.2 - Energ/L).tolist() 
energy1 = min(listi1) # we want to calcultate the energy nearest to zero
energy2 = min(listi2) # we want to calcultate the energy nearest to (-0.2)
ind1 = listi1.index(energy1)
ind2 = listi2.index(energy2)

for l in range(L+1):
    state_in_occu1 = rep_MSS_state(V[ind1])
    ent1 = Ent(state_in_occu1,L,l,L-l)
    Ent_entropies_for_particular_Eigenstate_1.append(ent1)
    
    state_in_occu2 = rep_MSS_state(V[ind2])
    ent2 = Ent(state_in_occu2,L,l,L-l)
    Ent_entropies_for_particular_Eigenstate_2.append(ent2)

plt.figure()
plt.scatter(np.arange(0,L+1,1),Ent_entropies_for_particular_Eigenstate_1)
plt.xlabel("Sub system size")
plt.ylabel("Entanglement entropy (for E/L nearest to 0)")
plt.title("Entangelement Entropy vs. Subsystem")
plt.show() 

plt.figure()
plt.scatter(np.arange(0,L+1,1),Ent_entropies_for_particular_Eigenstate_2)
plt.xlabel("Sub system size")
plt.ylabel("Entanglement entropy (for E/L nearest to -0.2)")
plt.title("Entangelement Entropy vs. Subsystem")
plt.show()  

sub_size = np.arange(0,L+1,1)
data_sub_size_E0_Eptminus2 = np.column_stack((sub_size,Ent_entropies_for_particular_Eigenstate_1,Ent_entropies_for_particular_Eigenstate_2))



           
#-------------------- Time Evolution of Entanglement entropy -----------------------  

dim_MSS = len(SRS_basis_MSS[1])

ini_state = np.zeros(dim_MSS)

ini_state[-1] = 1 # It is also in MSS basis, this is the |Psi_0> state 

ent_entropy_time_evolution = []

for t in range(0,6,1):
    Entropy = []
    psit = psi_t(t,ini_state).tolist() # psit is in MSS basis 
    Psit = rep_MSS_state(psit) # output is in the form  [[|n>,a_n],.....]
    for l in range(L+1):
        entrop = Ent(Psit, L,l, L-l)
        Entropy.append(entrop)
    ent_entropy_time_evolution.append(Entropy)    
    
plt.figure()    
plt.scatter(range(L+1),ent_entropy_time_evolution[0],marker = 'o',label ="$ t = 0.0 $")
plt.scatter(range(L+1),ent_entropy_time_evolution[1], marker = ',',label ="$ t = 1.0 $")
plt.scatter(range(L+1),ent_entropy_time_evolution[2], marker = 's',label ="$ t = 2.0 $")
plt.scatter(range(L+1),ent_entropy_time_evolution[3], marker = 'v',label ="$ t = 3.0 $")
plt.scatter(range(L+1),ent_entropy_time_evolution[4], marker = '.',label ="$ t = 4.0 $")
plt.scatter(range(L+1),ent_entropy_time_evolution[5], marker = '^',label ="$ t = 5.0 $")
plt.legend()
plt.title("Time evolution of Entanglement Entropy")
plt.show()   

