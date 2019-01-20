import numpy as np
from scipy import integrate
from pyapprox.configure_plots import *
def viral_ode_rhs(sol,time,params):
    T,I,V = sol
    beta,delta,p,c = params
    rhs = [-beta*T*V,beta*T*V-delta*I,p*I-c*V]
    return rhs

def viral_ode():
    y0 = [4e8, 0., 3.5e-1]
    params = [7.4e-5,3.4,7.9e-3,3.3]
    final_time = 8
    dt = 0.01
    time =  np.arange( 0, final_time+dt, dt )
    y = integrate.odeint( viral_ode_rhs, y0, time, args = (params,) )

    import matplotlib.pyplot as plt
    print y.max(axis=0)
    print y[0,:]
    plt.semilogy(time,y)
    plt.ylabel("Viral Titer (TCID/mL)")
    plt.xlabel("Time post infection (days)")
    plt.legend(["T","I","V"])
    plt.savefig('viral-ode-evolution.pdf')

def monomial_vandermonde(degree,x):
    assert x.ndim==1
    nsamples = x.shape[0]
    vand = np.ones((nsamples,1))
    for ii in range(1,degree+1):
        vand = np.hstack((vand,x[:,np.newaxis]**ii))
    return vand

def bivariate_gaussian_conditional_distribution(mean,covar,x1):
    """
    Compute X2 | X1 of a bivariate gaussian (X1,X2)
    """
    assert x1.ndim==1
    corrcoef = covar[0,1]/np.sqrt(covar[0,0]*covar[1,1])
    cond_mean = mean[1]+corrcoef*np.sqrt(covar[1,1]/covar[0,0])*(x1-mean[0])
    cond_var = (1-corrcoef**2)*covar[1,1]*np.ones_like(x1)
    return cond_mean, cond_var
    

import scipy.stats as ss
from pyapprox.visualization import get_meshgrid_function_data
def regression():
    np.random.seed(4)
    nsamples = 10
    mean = np.zeros(2)
    corrcoef = 0.5
    variance = np.array([1,0.1])
    covar = np.array([[variance[0],np.sqrt(variance.prod())*corrcoef],
                      [np.sqrt(variance.prod())*corrcoef,variance[1]]])
    normal_rv = ss.multivariate_normal(mean=mean,cov=covar)
    XY_joint_pdf = lambda x: normal_rv.pdf(x.T)
    samples = normal_rv.rvs(size=nsamples).T

    vand = monomial_vandermonde(degree,samples[0,:])
    coef = np.linalg.lstsq(vand,samples[1,:],rcond=None)[0]

    plot_limits=[-3,3,-2,2]; num_pts_1d=100; num_contour_levels=10
    X,Y,Z = get_meshgrid_function_data(XY_joint_pdf,plot_limits,num_pts_1d)
    vmin =0.1*Z.max()#Z.min()
    line=plt.contourf(
        X,Y,Z,levels=np.linspace(vmin,Z.max(),num_contour_levels),
        cmap=mpl.cm.coolwarm,)
    plt.plot(samples[0,:],samples[1,:],'ok')
    xx = np.linspace(plot_limits[0],plot_limits[1],101)
    vand = monomial_vandermonde(1,xx)
    plt.plot(xx,vand.dot(coef),'k',
             label=r'$E[\hat{Y}|X=x]=\hat{\beta_0}+\hat{\beta_1} x$')
    cond_mean,cond_var=bivariate_gaussian_conditional_distribution(mean,covar,xx)
    plt.plot(xx,cond_mean,'k--',label=r'$E[Y|X=x]=\beta_0+\beta_1 x$')
    plt.plot(xx,cond_mean+np.sqrt(cond_var),'r--')
    plt.plot(xx,cond_mean-np.sqrt(cond_var),'r--')
    plt.legend()
    plt.savefig('regression-example.pdf')

def interpolation():

    nsamples = 5
    samples = np.cos(np.linspace(0,np.pi,nsamples))
    function = lambda x: x*np.cos(2*np.pi*x)
    values  = function(samples) 
    vand = monomial_vandermonde(nsamples-1,samples)
    print np.linalg.cond(vand)
    coef = np.linalg.lstsq(vand,values,rcond=None)[0]
    xx = np.linspace(-1,1,101)
    vand = monomial_vandermonde(nsamples-1,xx)
    poly_vals = vand.dot(coef)
    true_vals = function(xx)
    plt.plot(xx,poly_vals,'k')
    plt.plot(samples,values,'ok')
    plt.plot(xx,true_vals,'--k')
    I = np.argmax(np.absolute(true_vals-poly_vals))
    plt.plot([xx[I],xx[I]],[min(true_vals[I],poly_vals[I]),
                max(true_vals[I],poly_vals[I])],'o-')
    plt.savefig('interpolation-example.pdf')
    #plt.show()

def stocks():
    M = np.array([[3,1],[2,2]])
    p = np.array([[100,120]]).T
    print M
    print p

    M_inv = np.linalg.inv(M)
    c = M_inv.dot(p)
    print M_inv
    print c

def housing_data_linear_least_squares(degrees=[1]):

    lb = 100; ub = 600
    b0 = 3.5e5
    b1 = 10
    b2 = 2
    #function = lambda xx: b0+b1*xx+b2*xx**2
    def function(xx):
        xx = (xx-lb)/(ub-lb)
        xx = 2*xx-1
        print xx
        return 1e6*np.exp(-(1-xx)**2)+b0

    nsamples = 7
    #samples = np.random.uniform(10,ub,nsamples)
    samples = np.linspace(lb,ub,nsamples)
    values = function(samples)
    
    xx = np.linspace(lb,ub,101)
    yy = function(xx)
    #plt.plot(xx,yy,'r--')

    fix,axs = plt.subplots(1,len(degrees),figsize=(8*len(degrees),6))
    for jj,degree in enumerate(degrees):
        vand = monomial_vandermonde(degree,samples)
        coef = np.linalg.lstsq(vand,values,rcond=None)[0]
        poly_vals_at_train = vand.dot(coef)
    
        vand = monomial_vandermonde(degree,xx)
        poly_vals = vand.dot(coef)
        for ii in range(nsamples):
            axs[jj].plot([samples[ii],samples[ii]],
                    [values[ii],poly_vals_at_train[ii]],'r')
            axs[jj].plot(xx,poly_vals,'k',)
        
        axs[jj].plot(samples,values,'bo')
        axs[jj].plot(xx,function(xx),'b-')

        axs[jj].set_ylabel('House Price (\$)')
        axs[jj].set_xlabel('Area ($m^2$)')
        axs[jj].ticklabel_format(axis='y',style='sci',scilimits=(0,0))

    plt.savefig('house-least-squares-degrees%s.pdf'%("".join(['-%s'%d for d in degrees])))
    
    plt.show()


def matrix_factorization_cost():
    n = np.logspace(1,5,10)

    plt.loglog(n,n**2,label='Substitution')
    plt.loglog(n,2./3*n**3,label='LU')
    plt.loglog(n,4./3*n**3, label='QR')

    plt.xlabel('$M$')
    plt.ylabel('Operations')
    plt.legend()

    plt.savefig('matrix-factorization-costs.pdf')
    #plt.show()

def housing_data_regression():

    b0 = 3.5e5
    b1 = 500
    function = lambda xx: b0+b1*xx

    nsamples = 100
    samples = np.random.uniform(100,600,nsamples)
    values = function(samples)
    noise = np.random.normal(0,2e4,nsamples)
    values += noise
    
    xx = np.linspace(100,600,101)
    yy = function(xx)
    #plt.plot(xx,yy,'r--')

    vand = monomial_vandermonde(1,samples)
    coef = np.linalg.lstsq(vand,values,rcond=None)[0]
    vand = monomial_vandermonde(1,xx)
    plt.plot(samples,values,'o')
    plt.plot(xx,vand.dot(coef),'k',)
    plt.ylabel('House Price (\$)')
    plt.xlabel('Area ($m^2$)')
    plt.savefig('house-regression.pdf')
    #plt.show()

def convergence():

    xx = np.logspace(0,5,10)
    for rr in [1,2,3]:
        plt.loglog(xx,xx**(-rr),label='$O(n^{-%d})$'%rr)
    xx = xx[:6]
    plt.loglog(xx,np.exp(-xx**(1)/10),label='$O(e^{-n/10})$')
    plt.legend()
    plt.savefig('convergence-rates.pdf')
    #plt.show()
    

if __name__=='__main__':
    #viral_ode()
    #regression()
    #interpolation()
    #stocks()
    #housing_data_linear_least_squares(degrees=[1,2])
    #housing_data_regression
    #convergence()
    matrix_factorization_cost()
    
    
