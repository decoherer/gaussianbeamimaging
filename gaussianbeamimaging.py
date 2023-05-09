import numpy as np
from numpy import pi,sqrt
import matplotlib.pyplot as plt
plt.rcParams['keymap.quit'] = ['ctrl+w','cmd+w','q','escape']

def gaussianbeamR(z,w0,λ): # radius of curvature of phase front at distance z from beam waist
    zr = pi*w0**2/λ
    return z*(1+zr**2/z**2)
def gaussianbeamw(z,w0,λ): # beam waist (minimum beam radius)
    zr = pi*w0**2/λ
    return w0*sqrt(1+(z/zr)**2)
def plotbeams(wi,zi,wo,zo,λ,zmargin=5): # wi,wo = beam waists in µm, zi,zo = distance to lens in mm, λ in nm
    xo = np.linspace(-zmargin,zo+zmargin,1001)
    xxo = np.linspace(0,zo,1001)
    xi = np.linspace(-zmargin+zo,zo+zi+zmargin,1001)
    xxi = np.linspace(zo,zo+zi,1001)
    yo = gaussianbeamw(z=xo,w0=wo,λ=λ)
    yyo = gaussianbeamw(z=xxo,w0=wo,λ=λ)
    yi = gaussianbeamw(z=zo+zi-xi,w0=wi,λ=λ)
    yyi = gaussianbeamw(z=zo+zi-xxi,w0=wi,λ=λ)
    plt.plot(xo,+yo,c='g',ls='dotted')
    plt.plot(xo,-yo,c='g',ls='dotted')
    plt.plot(xxo,+yyo,c='g')
    plt.plot(xxo,-yyo,c='g')
    plt.plot(xi,+yi,c='b',ls='dotted')
    plt.plot(xi,-yi,c='b',ls='dotted')
    plt.plot(xxi,+yyi,c='b')
    plt.plot(xxi,-yyi,c='b')
    plt.xlabel('mm')
    plt.ylabel('µm')
    plt.show()
def createtestbeams(zi=30,wi=40,d=45,λ=1000):
    # create a test case where image beam waist is wi µm and zi mm from lens
    # with distance d between image and object beam waist
    w = gaussianbeamw(z=zi,w0=wi,λ=λ) # beam radius at lens
    zo = d-zi
    woest = wi*zo/zi # print('wo initial estimate',woest)
    wos = np.linspace(woest-5,woest+5,1001)
    ws = gaussianbeamw(z=zo,w0=wos,λ=λ)
    wo = np.interp(w,ws[::-1],wos[::-1])
    f = 1/(1/gaussianbeamR(z=zo,w0=wo,λ=λ)+1/gaussianbeamR(z=zi,w0=wi,λ=λ))
    # print('wi',wi)
    # print('wo',wo)
    # print('zi',zi)
    # print('zo',zo)
    # print('r',pi/λ*gaussianbeamw(z=zi,w0=wi,λ=λ)**2)
    # print('ri',pi/λ*wi**2)
    # print('ro',pi/λ*wo**2)
    # print('fo',gaussianbeamR(z=zo,w0=wo,λ=λ))
    # print('fi',gaussianbeamR(z=zi,w0=wi,λ=λ))
    # print('f',f)
    return wo,f # object waist, lens focal length
def raymagnification(d,f,λ): # µm,mm,mm,nm units
    # zi + zo = d, 1/zi + 1/zo = 1/f
    # α≡f/zo, β≡f/zi, γ≡f/d
    # 1/α + 1/β = 1/γ, α + β = 1 → α²-α+γ = 0
    # α,β = ½-√(¼-γ),½+√(¼-γ)
    zo = f/(0.5+sqrt(0.25-f/d))
    zi = f/(0.5-sqrt(0.25-f/d))
    assert np.allclose(d,zo+zi)
    return zi/zo
def gaussianmagnification(wi,d,f,λ,plot=False): # µm,mm,mm,nm units
    # returns object beam waist ωo in µm given:
    #   wi = imaged beam waist ωi in µm
    #   d = distance from object waist to image waist in mm
    #   f = focal length of imaging lens in mm
    #   λ = wavelength in nm
    zi = np.linspace(d/2,d-f,1001)      # zi = distance from image waist to lens # assume M>1 therefore zo<zi
    ri = wi**2 * pi/λ                # ri = zR for image beam
    fi = zi*(1+ri**2/zi**2)             # fi = ROC of image beam at lens
    fo = 1/(1/f-1/fi)                   # fo = ROC of object beam at lens
    r = ri*(1+zi**2/ri**2)              # r = zR for beam with waist equal to beam radius at the lens
    zof = r**2/fo/(1+r**2/fo**2)        # zo via 1/f = 1/fi + 1/fo
    zod = d-zi                          # zo via d = zi + zo
    zo = np.interp(0,zof-zod,zod)       # zo = distance from object waist to lens
    ro = np.interp(0,zof-zod,zo*fo/r)   # ro = zR for object beam
    wo = sqrt(ro*λ/pi)            # wo = object beam waist ωo
    na = np.interp(0,zof-zod,1e-3*sqrt(r*λ/pi)/zo) # minimum necessary NA of lens
    # print('zi',np.interp(0,zof-zod,zi),'zo',zo,'ro',ro,'wo',wo)
    if plot:
        plotbeams(wi=wi,zi=d-zo,wo=wo,zo=zo,λ=λ)
    return wi/wo,zo,na

if __name__ == '__main__':

    # create a test setup by finding input waist and lens given output distance from lens, output waist, distance from input to output, and wavelength
    wo,f = createtestbeams(zi=30,wi=40,d=45,λ=1000)
    print('wo',wo,'f',f) # wo 19.791304557290363 f 10.137453578080027
    # plot the test setup
    plotbeams(wi=40,zi=30,wo=19.79130616621962,zo=15,λ=1000)

    # find the magnification given only output waist, distance from input to output, local length, and wavelength
    M,zo,na = gaussianmagnification(wi=40,d=45,f=10.137453578080027,λ=1000,plot=True)
    # M,zo,na = gaussianmagnification(wi=300,d=45,f=10,λ=1000,plot=True)
    print('M',M,'zo',zo,'na',na) # magnification, distance from output to lens, minimum NA of lens

    # compare to magnification calculated using ray optics
    Mray = raymagnification(d=45,f=10.137453578080027,λ=1000)
    print('Mray',Mray)
