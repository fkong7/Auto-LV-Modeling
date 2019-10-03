import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D


def readsample(Npts, Nspl):
    xspl = np.zeros((Npts,Nspl))
    yspl = np.zeros((Npts,Nspl))
    zspl = np.zeros((Npts,Nspl))

    # Read the rest
    for i in range(Nspl):
        path = "Clip_"+str(i)+'-mesh-complete'
        file = path+'/'+'N_mesh-complete.exterior'+'.txt'
        print(file)
        coord = np.loadtxt(file)
        xspl[:,i] = coord[:,0]
        yspl[:,i] = coord[:,1]
        zspl[:,i] = coord[:,2]

    return xspl, yspl, zspl 

def myinterpolate(xspl,yspl,zspl,Ntime,uu,ufine):
    tck, u = interpolate.splprep([xspl,yspl,zspl], u=uu,s=0,per=True)
    x,y,z=interpolate.splev(ufine, tck)

    return x, y, z

if __name__=='__main__':

    Npts  = 4754 # Number of points
    Nspl  = 10   # Number of samples
    Nint  = 11   # Number of points used in interpolation
    Ntime = 3001 # Number of time steps

    # Read sample data
    xtmp, ytmp, ztmp = readsample(Npts,Nspl)
    xspl = np.zeros((Npts,Nint))
    yspl = np.zeros((Npts,Nint))
    zspl = np.zeros((Npts,Nint))
    for i in range(Nint):
        xspl[:,i], yspl[:,i], zspl[:,i] = xtmp[:,-1], ytmp[:,-1], ztmp[:,-1]

    for i in range(1):#range(np.int(np.floor(Nint/Nspl))):
        for j in range(Nspl):
            k = i*Nspl+j+1
            print(k)
            xspl[:,k] = xtmp[:,j]
            yspl[:,k] = ytmp[:,j]
            zspl[:,k] = ztmp[:,j]

    xfine = np.zeros((Npts,Ntime))
    yfine = np.zeros((Npts,Ntime))
    zfine = np.zeros((Npts,Ntime))
    u = np.linspace(0,1,Nint)
    ufine = np.linspace(0,1,Ntime)
    for i in range(Npts):
        xfine[i,:],yfine[i,:],zfine[i,:] = myinterpolate(xspl[i,:],yspl[i,:],zspl[i,:],Ntime,u,ufine)

    path = './Interpolated'
    for i in range(Ntime):
        coord = np.stack((xfine[:,i], yfine[:,i], zfine[:,i]),axis=1)

        fname = "mesh-complete.exterior_"+format(i, '05d')
        fout = path+'/'+fname+'.txt'
        np.savetxt(fout,coord)


    # fig = plt.figure()
    # ax3d = fig.add_subplot(111,projection='3d')
    # ax3d.plot(xspl[1,:],yspl[1,:],zspl[1,:],'bo')
    # ax3d.plot(xfine[1,:], yfine[1,:], zfine[1,:], 'g')
    # fig.show()
    
    
