import fastfilters
import numpy
import math
import vigra

def extractChannels(data,usedChannels=(0,)):
    
    if data.ndim == 2:
        d =data[:,:,None]
    elif data.ndim == 3:
        d = data
    else:
        raise TypeError("Invalid input data")

    d = d[:,:,usedChannels]

    if d.ndim == 2:
        d =d[:,:,None]
    elif d.ndim == 3:
        d = d
    else:
        raise TypeError("Invalid input data")

    return d


def getScale(target, presmoothed):
    return math.sqrt(target**2 - presmoothed**2)

class ConvolutionFeatures:
    def __init__(self, sigmas=(1.0, 2.0, 4.0, 8.0), usedChannels=(0,)):
        self.sigmas = sigmas
        self.usedChannels = usedChannels

        maxSigma = max(sigmas)
        maxOrder = 2
        r = int(round(3.0*maxSigma* + 0.5*2))

        # r of structure tensor
        rD = int(round(3.0*maxSigma*0.3 + 0.5))
        rG = int(round(3.0*maxSigma*0.7 ))
        rSt = rD+rG

        self.__halo = (max(r,rSt),)*2

    @property
    def halo(self):
        return self.__halo

    @property
    def n_features(self):
        nEvFeat = 2
        nScalarFeat =  3
        nEv = 2
        nChannels = len(self.usedChannels)
        return  len(self.sigmas) * (nEvFeat*nEv + nScalarFeat) * nChannels


    def __call__(self, dataIn, slicing, featureArray):
        fIndex = 0
        dataIn = numpy.require(dataIn,'float32').squeeze()

        dataWithChannel = extractChannels(dataIn, self.usedChannels)

        slicingEv = slicing + [slice(0,2)]

        for c in range(dataWithChannel.shape[2]):

            data = dataWithChannel[:,:,c]

            # pre-smoothed
            sigmaPre = self.sigmas[0]/2.0
            preS = fastfilters.gaussianSmoothing(data, sigmaPre)

            for sigma in self.sigmas:

                neededScale = getScale(target=sigma, presmoothed=sigmaPre)
                preS = fastfilters.gaussianSmoothing(preS, neededScale)
                sigmaPre = sigma

                featureArray[:,:,fIndex] = preS[slicing]
                fIndex += 1

                featureArray[:,:,fIndex] = fastfilters.laplacianOfGaussian(preS, neededScale)[slicing]
                fIndex += 1

                featureArray[:,:,fIndex] = fastfilters.gaussianGradientMagnitude(preS, neededScale)[slicing]
                fIndex += 1


                featureArray[:,:,fIndex:fIndex+2] = fastfilters.hessianOfGaussianEigenvalues(preS, neededScale)[slicingEv]
                fIndex += 2

                
                #print("array shape",featureArray[:,:,:,fIndex:fIndex+3].shape)
                feat = fastfilters.structureTensorEigenvalues(preS, float(sigma)*0.3, float(sigma)*0.7)[slicingEv]
                #print("feat  shape",feat.shape)
                featureArray[:,:,fIndex:fIndex+2] = feat
                fIndex += 2

        assert fIndex == self.n_features