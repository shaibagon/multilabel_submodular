#include <mex.h>
#include "graph.h"




/*
 *
 * Discrete optimization of the following multi-label functional
 *
 *  x = argmin \sum_i D_i(x_i) + \sum_ij w_ij V_k (x_i, x_j)
 *
 * Usage:
 *
 *  x = MultiLabelSubModular_mex(D, W, V)
 *
 * Inputs:
 *  D   - unary term, LxN double matrix
 *  W   - sparse _complex_ NxN matrix with 
 *          real part w_ij - the wieght of the i-j interaction
 *          imaginary part - integer index into matrices V (matlab's 1-based index)
 *        W should be symmetric (redundent representation for speed computation)
 *  V   - LxLxK double array with the V_k Monge interaction matrix
 *
 * Output:
 *  x   - Globally optimal labeling
 *
 *
 * compile
 * >> mex -O -largeArrayDims -DNDEBUG graph.cpp maxflow.cpp MultiLabelSubModular_mex.cpp -output MultiLabelSubModular_mex
 */

void my_err_function(const char* msg) {
    mexErrMsgTxt(msg);
}

// inputs
enum {
    iD = 0,
    iW,
    iV,
    nI
};

// outputs
enum {
    oX = 0,
    nO
};


void
mexFunction(
    int nout,
    mxArray* pout[],
    int nin,
    const mxArray* pin[])
{

    /****************************************************************
     * Check Inputs
     */
    if ( nin != nI )
        mexErrMsgIdAndTxt("MultiLabelSubModular_mex:n_inputs", "Must have %d inputs", nI);
    
    if ( nout != nO )
        mexErrMsgIdAndTxt("MultiLabelSubModular_mex:n_outputs", "Must have %d outputs", nO);
    
    // check unary term
    if ( !mxIsDouble(pin[iD]) || mxIsComplex(pin[iD]) || 
            mxIsSparse(pin[iD]) || mxGetNumberOfDimensions(pin[iD])!=2 )
        mexErrMsgIdAndTxt("MultiLabelSubModular_mex:D_mat", "D must be 2D double real matrix");
    
    mwSize L = mxGetM(pin[iD]);
    mwSize N = mxGetN(pin[iD]);
    double* pD = mxGetPr(pin[iD]);
    
    // check W
    if ( !mxIsDouble(pin[iW]) || !mxIsComplex(pin[iW]) || ! mxIsSparse(pin[iW]) ||
          mxGetNumberOfDimensions(pin[iW])!=2 || mxGetM(pin[iW]) != N || mxGetN(pin[iW]) != N )
        mexErrMsgIdAndTxt("MultiLabelSubModular_mex:W_mat", 
                "W must be sparse %dx%d matrix imaginary matrix", N, N);

    // check V
    if ( !mxIsDouble(pin[iV]) || mxIsComplex(pin[iV]) || mxIsSparse(pin[iV]) ||
            mxGetNumberOfDimensions(pin[iV]) < 2 || mxGetNumberOfDimensions(pin[iV]) > 3) 
        mexErrMsgIdAndTxt("MultiLabelSubModular_mex:V_arr","V array must be real 3D double array");
    
    const mwSize* dim_v = mxGetDimensions(pin[iV]);
    
    // how many V matrices are there?
    mwIndex NumV(1);
    if ( mxGetNumberOfDimensions(pin[iV]) == 3 ) {
        NumV = dim_v[2];
    }
    
    if ( dim_v[0] != L || dim_v[1] != L )
        mexErrMsgIdAndTxt("MultiLabelSubModular_mex:V_arr_size", "V must be %dx%dxK array", L, L);
    
    double* pV = mxGetPr(pin[iV]);
    
    
    /****************************************************************
     * construct the graph
     */

    double*  pWij= mxGetPr(pin[iW]);
    double*  pVi = mxGetPi(pin[iW]);
	mwIndex* pir = mxGetIr(pin[iW]);
	mwIndex* pjc = mxGetJc(pin[iW]);
    
                    
    mwSize E = pjc[N]; // number of non zeros is the last element in Jc 
    
    
    const double INF(1e100); // or other very large number
    
    
    // allocate space for graph
    typedef Graph<double, double, double> dGraph; 
    dGraph* gp = 
            new dGraph( N*(L-1),/* number of nodes excluding s/t */
                        N*(L-2) + E*(L-1)*(L-1),/* number of edges */
                        my_err_function /* error function handle */ );

        
    gp->add_node(N*(L-1)); // add all nodes
    
    /****************************************************************
     * Unary term + all associated edges
     */   
    mwSize edgeCounter(0);
    
    
    for ( mwIndex ii(0) ; ii < N ; ii++ ) {
        
        for ( mwIndex li(0) ; li < L-1 ; li++ ) {
            
            double qrk(0);
            
            // all neighbors of ii - a column in W
            for ( mwIndex jj =  pjc[ii] ; // starting row index
					jj < pjc[ii+1]  ; // stopping row index
					jj++)  {
                
                double wij = pWij[jj];
                mxAssert( wij > 0 , "weight wij must be positive");
                
                mwIndex vi = static_cast<mwIndex>(pVi[jj]);
                mxAssert( vi > 0 && vi <= NumV, "index into v out of range" );
                vi--; // convert to 0-based index
                
                qrk += wij * ( pV[L*L*vi + li] // grr'(k,1)
                        + pV[L*L*vi + li + L*(L-1)] // grr'(k,|K|)
                        - pV[L*L*vi + li + 1] // grr'(k+1,1)
                        - pV[L*L*vi + li + 1 + L*(L-1)] ); // grr'(k+1,|K|) 
            }
            
            qrk = qrk / 2.0;
                  
            qrk += ( pD[ii*L + li] - pD[ii*L + li + 1 ] );
            
            if ( qrk > 0 ) {
                gp -> add_tweights( ii*(L-1) + li, qrk, 0);
            } else {
                gp -> add_tweights( ii*(L-1) + li, 0, -qrk);
            }
            
            // adding between states edges
            if ( li < L - 2 ) {
                gp -> add_edge( ii*(L-1) + li, ii*(L-1) + li + 1, 0, INF );
                edgeCounter++;
            }
        }
    }
    mxAssert( edgeCounter == N*(L-2) , "wrong number of constraint edges");
    edgeCounter = 0;
    

    /****************************************************************
     * pair-wise terms
     */
    
        
    for ( mwIndex ii(0) ; ii < N ; ii++ ) {
            for ( mwIndex ri =  pjc[ii] ; // starting row index
					ri < pjc[ii+1]  ; // stopping row index
					ri++)  {
         
            mwIndex jj = pir[ri];        
        
            double wij = pWij[ri];
            mxAssert( wij > 0 , "weight wij must be positive");
                
            mwIndex vi = static_cast<mwIndex>(pVi[ri]);
            mxAssert( vi > 0 && vi <= NumV, "index into v out of range" );
            vi--; // convert to 0-based index
        
        
        
            for ( mwIndex li(0) ; li < L-1 ; li++ ) {
                for ( mwIndex lj(0); lj < L-1 ; lj++ ) {
                
                    double arr(0);
                
                    arr = wij * ( pV[L*L*vi + li + L*lj] // grr'(k,k')
                           + pV[L*L*vi + li + 1 + L*(lj+1)] // grr'(k+1,k'+1)
                           - pV[L*L*vi + li + 1 + L*lj] // grr'(k+1,k')
                           - pV[L*L*vi + li + L*(lj+1)] );// grr'(k,k'+1)
                
                    arr = -arr/2;
                    mxAssert( arr > 0 , "non submodular term?");
                
                    gp -> add_edge( ii*(L-1) + li, jj*(L-1) + lj, arr, arr);
                    edgeCounter++;
                }
            }

        }
    }
    
    mxAssert( edgeCounter == E*(L-1)*(L-1), "wrong number of pair-wise edges");
    
    /****************************************************************
     * optimize!
     */
    double e = gp->maxflow();
    

    /****************************************************************
     * read results
     */
//    pout[oE] = mxCreateDoubleScalar(e); // output the flow

    
    pout[oX] = mxCreateDoubleMatrix(1, N, mxREAL);
    double* pX = mxGetPr(pout[oX]);
    
    for ( mwIndex ii(0) ; ii < N ; ii++ ) {
        
        pX[ii] = static_cast<double>(L);
        
        for ( mwIndex li(0) ; li < L-1 ; li++ ) {
            if ( gp->what_segment( ii*(L-1) + li ) == dGraph::SINK ) {
                pX[ii] = static_cast<double>(li+1);
                break;
            }
        }            
        
    }
  

/*    
    pout[oX] = mxCreateDoubleMatrix(N, L-1, mxREAL);
    double* pX = mxGetPr(pout[oX]);
    
    for ( mwIndex li(0) ; li < L-1 ; li++ ) {
        for ( mwIndex ii(0) ; ii < N ; ii++ ) {
        
            pX[ii + N*li] = static_cast<double>(
                    gp->what_segment( ii*(L-1) + li ) == dGraph::SINK
                    );
            
        }            
        
    }
*/    
    delete gp; // clear memory
}
