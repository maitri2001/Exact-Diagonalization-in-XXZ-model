——————————————————Instructions to Run Code for Exact Diagonalization in XXZ model mapped to hard-core boson—————————————————

 ## Setting the Parameters

 In the source file, various model parameters are defined just in the beginning. These are,

 L (length of the 1d chain), 
 Delta (relative strength of onsite term compared to hopping), and 
 lamb (relative weight of next-to-nearest neighbour interaction term)

 Set these values to whatever you want.


 ## -----------------Running the source file-------------------------------------------------------

Now, if you simply run the source file - "exact_diagonalization_source_code.py " , all the functions defined within it will be usable. In the source file, there some code written for doing some calculations, but you can comment those portions out, but keeping only the function definitions.

This file has three parts the first part "MAIN CODE AND SOME DATA COLLECTION" has all the defined functions and there are two three data saving part which are commented(enclosed within triple quotes). 

In the second part of the code " PLOTS" it is commented but when needed it can be uncommented (removing the triple quotes) to store all the data required and plot them directly. 

In the third part "Making the plots from the stored data" basically all the data which were already stored in the src folder are loaded and been plotted for different system sizes or for different variables as needed which is also commented. Uncomment this third part to produce all the plots which are attached in my report. 


## ------------------------------------------------Running the example code---------------------------------------------
Run the "L8.py" file in the example folder as the example code for system size L=8 and get all the plots for that system size


## -----------------Getting the Hamiltonian--------------------------------------------------------

1. The full 2^L x 2^L Hamiltonian matrix is constructed by  H_full_basis(L). For safety, keep the calculation of eigenvalues of H_full_basis(L) commented in the source file, since for even L=14, it is a very expensive task.

2. The Hamiltonian in the N-particle sector is constructed by HN(L,N); The basis in the N-particle sector is constructed by Basis(N) and Basis2(N), which contain the binary string representation and integer representation respectively.

3. The Hamiltonian in the MSS sector is constructed by HMSS(L); The basis for MSS is given in terms of Super Representative States/SRS are stored in SRS_basis_MSS[1]. dim_MSS gives the dimension of the MSS sector which is just len(SRS_basis_MSS[1]). Diagonalize this Hamiltonian to get the eigenvalues and eigenvectors in the MSS sector. 

4. Eigenvalues in the MSS sector are stored in the array Energ, and the eigenvectors are stored in the 2d array V, whose rows are the desired eigenvectors. On the other hand, transpose of V, U has eigenvectors as columns. (Don’t comment out the lines where Energ, U are calculated)


## -----------------Changing basis-----------------------------------------------------------------

1. A general state in the MSS sector, stateMSS, can be represented as an array with dim_MSS elements, which encode the weights corresponding to each basis state in the MSS. This can be converted to a form [[string,weight],…] in occupation number basis by using the function rep_MSS_state(stateMSS). 

2. On the other hand, a state of the form  state=[[string,weight],…] can be converted to an array with 2^L elements, where the number in the nth position is weight corresponding to the string representation of the decimal n. This can be done using the function colmn_full_basis(state).


## ---------------Exact Time evolution---------------------------------------------------------------

Exact time evolution on a state defined in the MSS basis, can be obtained using the function psi_t(t,stateMSS).


## ---------------Entanglement Entropy----------------------------------------------------------------------------

For a state defined in the form, state = [[string,weight],…], the entanglement entropy can be calculated using Ent(state,L,L1,L2) , where L1 and L2 are the sizes of the subsystem and L1+L2 = L.


##--------------- Approximate Time evolution--------------------------------------------------------------

Approximate time evolution for a state given as array with 2^L elements, that is, a state in the full Hilbert space, can be done by suitable application of the two-spin rotation operator. The two-spin rotation operator for the two sites l,m, is implemented through act_u(state,L,l,m,eps), where eps in the time step. For definition of the two-spin rotation operator see Eq.(47) of J. H. Jung, J. D. Noh, Guide to Exact Diagonalization Study of Quantum Thermalization, Journal of the Korean Physical Society (2020).

