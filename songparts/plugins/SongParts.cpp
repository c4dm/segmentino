/* -*- c-basic-offset: 4 indent-tabs-mode: nil -*-  vi:set ts=8 sts=4 sw=4: */

/*
    QM Vamp Plugin Set

    Centre for Digital Music, Queen Mary, University of London.

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version.  See the file
    COPYING included with this distribution for more information.
*/

#include "SongParts.h"

#include <base/Window.h>
#include <dsp/onsets/DetectionFunction.h>
#include <dsp/onsets/PeakPicking.h>
#include <dsp/transforms/FFT.h>
#include <dsp/tempotracking/TempoTrackV2.h>
#include <dsp/tempotracking/DownBeat.h>
#include <chromamethods.h>
#include <maths/MathUtilities.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/math/distributions/normal.hpp>
#include "armadillo"
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>

#include <vamp-sdk/Plugin.h>

using namespace boost::numeric;
using namespace arma;
using std::string;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;


#ifndef __GNUC__
#include <alloca.h>
#endif


// Result Struct
typedef struct Part {
    int n;
    vector<unsigned> indices;
    string letter;
    unsigned value;
    int level;
    int nInd;
}Part;


/* ------------------------------------ */
/* ----- BEAT DETECTOR CLASS ---------- */
/* ------------------------------------ */

class BeatTrackerData
{
    /* --- ATTRIBUTES --- */
public:
    DFConfig dfConfig;
    DetectionFunction *df;
    DownBeat *downBeat;
    vector<double> dfOutput;
    Vamp::RealTime origin;
    
    
    /* --- METHODS --- */
    
    /* --- Constructor --- */
public:
    BeatTrackerData(float rate, const DFConfig &config) : dfConfig(config) {
	
        df = new DetectionFunction(config);
        // decimation factor aims at resampling to c. 3KHz; must be power of 2
        int factor = MathUtilities::nextPowerOfTwo(rate / 3000);
        // std::cerr << "BeatTrackerData: factor = " << factor << std::endl;
        downBeat = new DownBeat(rate, factor, config.stepSize);
    }
    
    /* --- Desctructor --- */
    ~BeatTrackerData() {
	delete df;
        delete downBeat;
    }
    
    void reset() {
        delete df;
        df = new DetectionFunction(dfConfig);
        dfOutput.clear();
        downBeat->resetAudioBuffer();
        origin = Vamp::RealTime::zeroTime;
    }
};


/* --------------------------------------- */
/* ----- CHROMA EXTRACTOR CLASS ---------- */
/* --------------------------------------- */

class ChromaData
{
    
     /* --- ATTRIBUTES --- */
    
public:
    int frameCount;
    int nBPS;
    Vamp::Plugin::FeatureList logSpectrum;
    size_t blockSize;
    int lengthOfNoteIndex;
    vector<float> meanTunings;
    vector<float> localTunings;
    float whitening;
    float preset;
    float useNNLS;
    vector<float> localTuning;
    vector<float> kernelValue;
    vector<int> kernelFftIndex;
    vector<int> kernelNoteIndex;
    float *dict;
    bool tuneLocal;
    float doNormalizeChroma;
    float rollon;
    float s;
    vector<float> hw;
    vector<float> sinvalues;
    vector<float> cosvalues;
    Window<float> window;
    FFTReal fft;
    size_t inputSampleRate;
    
    /* --- METHODS --- */
    
    /* --- Constructor --- */
    
public:
    ChromaData(float inputSampleRate, size_t block_size) :
        frameCount(0),
        nBPS(3),
        logSpectrum(0),
        blockSize(0),
        lengthOfNoteIndex(0),
        meanTunings(0),
        localTunings(0),
        whitening(1.0),
        preset(0.0),
        useNNLS(1.0),
        localTuning(0.0),
        kernelValue(0),
        kernelFftIndex(0),
        kernelNoteIndex(0),
        dict(0),
        tuneLocal(0.0),
        doNormalizeChroma(0),
        rollon(0.0),
    	s(0.7),
    	sinvalues(0),
    	cosvalues(0),
    	window(HanningWindow, block_size),
    	fft(block_size),
    	inputSampleRate(inputSampleRate)
    {
        // make the *note* dictionary matrix
        dict = new float[nNote * 84];
        for (int i = 0; i < nNote * 84; ++i) dict[i] = 0.0;
        blockSize = block_size;
    }
    
    
    /* --- Desctructor --- */
    
    ~ChromaData() {
        delete [] dict;
    }
    
    /* --- Public Methods --- */
    
    void reset() {
        frameCount = 0;
        logSpectrum.clear();
        for (int iBPS = 0; iBPS < 3; ++iBPS) {
            meanTunings[iBPS] = 0;
            localTunings[iBPS] = 0;
        }
        localTuning.clear();
    }
    
    void baseProcess(float *inputBuffers, Vamp::RealTime timestamp)
    {   
        	
        frameCount++;   
        float *magnitude = new float[blockSize/2];
        double *fftReal = new double[blockSize];
        double *fftImag = new double[blockSize];

        // FFTReal wants doubles, so we need to make a local copy of inputBuffers
        double *inputBuffersDouble = new double[blockSize];
        for (size_t i = 0; i < blockSize; i++) inputBuffersDouble[i] = inputBuffers[i];
        
        fft.process(false, inputBuffersDouble, fftReal, fftImag);
        
        float energysum = 0;
        // make magnitude
        float maxmag = -10000;
        for (int iBin = 0; iBin < static_cast<int>(blockSize/2); iBin++) {
            magnitude[iBin] = sqrt(fftReal[iBin] * fftReal[iBin] + 
                                   fftImag[iBin] * fftImag[iBin]);
            if (magnitude[iBin]>blockSize*1.0) magnitude[iBin] = blockSize; 
            // a valid audio signal (between -1 and 1) should not be limited here.
            if (maxmag < magnitude[iBin]) maxmag = magnitude[iBin];
            if (rollon > 0) {
                energysum += pow(magnitude[iBin],2);
            }
        }
    
        float cumenergy = 0;
        if (rollon > 0) {
            for (int iBin = 2; iBin < static_cast<int>(blockSize/2); iBin++) {
                cumenergy +=  pow(magnitude[iBin],2);
                if (cumenergy < energysum * rollon / 100) magnitude[iBin-2] = 0;
                else break;
            }
        }
    
        if (maxmag < 2) {
            // cerr << "timestamp " << timestamp << ": very low magnitude, setting magnitude to all zeros" << endl;
            for (int iBin = 0; iBin < static_cast<int>(blockSize/2); iBin++) {
                magnitude[iBin] = 0;
            }
        }
        
        // cerr << magnitude[200] << endl;
        
        // note magnitude mapping using pre-calculated matrix
        float *nm  = new float[nNote]; // note magnitude
        for (int iNote = 0; iNote < nNote; iNote++) {
            nm[iNote] = 0; // initialise as 0
        }
        int binCount = 0;
        for (vector<float>::iterator it = kernelValue.begin(); it != kernelValue.end(); ++it) {
            nm[kernelNoteIndex[binCount]] += magnitude[kernelFftIndex[binCount]] * kernelValue[binCount];
            binCount++;  
        }
    
        float one_over_N = 1.0/frameCount;
        // update means of complex tuning variables
        for (int iBPS = 0; iBPS < nBPS; ++iBPS) meanTunings[iBPS] *= float(frameCount-1)*one_over_N;
    
        for (int iTone = 0; iTone < round(nNote*0.62/nBPS)*nBPS+1; iTone = iTone + nBPS) {
            for (int iBPS = 0; iBPS < nBPS; ++iBPS) meanTunings[iBPS] += nm[iTone + iBPS]*one_over_N;
            float ratioOld = 0.997;
            for (int iBPS = 0; iBPS < nBPS; ++iBPS) {
                localTunings[iBPS] *= ratioOld; 
                localTunings[iBPS] += nm[iTone + iBPS] * (1 - ratioOld);
            }
        }
    
        float localTuningImag = 0;
        float localTuningReal = 0;
        for (int iBPS = 0; iBPS < nBPS; ++iBPS) {
            localTuningReal += localTunings[iBPS] * cosvalues[iBPS];
            localTuningImag += localTunings[iBPS] * sinvalues[iBPS];
        }
    
        float normalisedtuning = atan2(localTuningImag, localTuningReal)/(2*M_PI);
        localTuning.push_back(normalisedtuning);
    
        Vamp::Plugin::Feature f1; // logfreqspec
        f1.hasTimestamp = true;
        f1.timestamp = timestamp;
        for (int iNote = 0; iNote < nNote; iNote++) {
            f1.values.push_back(nm[iNote]);
        }
    
        // deletes
        delete[] inputBuffersDouble;
        delete[] magnitude;
        delete[] fftReal;
        delete[] fftImag;
        delete[] nm;
    
        logSpectrum.push_back(f1); // remember note magnitude
    }
    
    bool initialise()
    {
        dictionaryMatrix(dict, s);
	
    	// make things for tuning estimation
    	for (int iBPS = 0; iBPS < nBPS; ++iBPS) {
            sinvalues.push_back(sin(2*M_PI*(iBPS*1.0/nBPS)));
            cosvalues.push_back(cos(2*M_PI*(iBPS*1.0/nBPS)));
        }
    
	
    	// make hamming window of length 1/2 octave
    	int hamwinlength = nBPS * 6 + 1;
        float hamwinsum = 0;
        for (int i = 0; i < hamwinlength; ++i) { 
            hw.push_back(0.54 - 0.46 * cos((2*M_PI*i)/(hamwinlength-1)));    
            hamwinsum += 0.54 - 0.46 * cos((2*M_PI*i)/(hamwinlength-1));
        }
        for (int i = 0; i < hamwinlength; ++i) hw[i] = hw[i] / hamwinsum;
    
    
        // initialise the tuning
        for (int iBPS = 0; iBPS < nBPS; ++iBPS) {
            meanTunings.push_back(0);
            localTunings.push_back(0);
        }
	
        blockSize = blockSize;
        frameCount = 0;
        int tempn = nNote * blockSize/2;
        // cerr << "length of tempkernel : " <<  tempn << endl;
        float *tempkernel;

        tempkernel = new float[tempn];

        logFreqMatrix(inputSampleRate, blockSize, tempkernel);
        kernelValue.clear();
        kernelFftIndex.clear();
        kernelNoteIndex.clear();
        int countNonzero = 0;
        for (int iNote = 0; iNote < nNote; ++iNote) { 
            // I don't know if this is wise: manually making a sparse matrix
            for (int iFFT = 0; iFFT < static_cast<int>(blockSize/2); ++iFFT) {
                if (tempkernel[iFFT + blockSize/2 * iNote] > 0) {
                    kernelValue.push_back(tempkernel[iFFT + blockSize/2 * iNote]);
                    if (tempkernel[iFFT + blockSize/2 * iNote] > 0) {
                        countNonzero++;
                    }
                    kernelFftIndex.push_back(iFFT);
                    kernelNoteIndex.push_back(iNote);				
                }
            }
        }
        delete [] tempkernel;
    }    
};


/* --------------------------------- */
/* ----- SONG PARTITIONER ---------- */
/* --------------------------------- */


/* --- ATTRIBUTES --- */

float SongPartitioner::m_stepSecs = 0.01161;            // 512 samples at 44100
size_t SongPartitioner::m_chromaFramesizeFactor = 16;   // 16 times as long as beat tracker's
size_t SongPartitioner::m_chromaStepsizeFactor = 4;     // 4 times as long as beat tracker's


/* --- METHODS --- */

/* --- Constructor --- */
SongPartitioner::SongPartitioner(float inputSampleRate) :
    Vamp::Plugin(inputSampleRate),
    m_d(0),
    m_bpb(4),
    m_pluginFrameCount(0)
{
}


/* --- Desctructor --- */
SongPartitioner::~SongPartitioner()
{
    delete m_d;
}


/* --- Methods --- */
string SongPartitioner::getIdentifier() const
{
    return "qm-songpartitioner";
}

string SongPartitioner::getName() const
{
    return "Song Partitioner";
}

string SongPartitioner::getDescription() const
{
    return "Estimate contiguous segments pertaining to song parts such as verse and chorus.";
}

string SongPartitioner::getMaker() const
{
    return "Queen Mary, University of London";
}

int SongPartitioner::getPluginVersion() const
{
    return 2;
}

string SongPartitioner::getCopyright() const
{
    return "Plugin by Matthew Davies, Christian Landone, Chris Cannam, Matthias Mauch and Massimiliano Zanoni  Copyright (c) 2006-2012 QMUL - All Rights Reserved";
}

SongPartitioner::ParameterList SongPartitioner::getParameterDescriptors() const
{
    ParameterList list;

    ParameterDescriptor desc;

    desc.identifier = "bpb";
    desc.name = "Beats per Bar";
    desc.description = "The number of beats in each bar";
    desc.minValue = 2;
    desc.maxValue = 16;
    desc.defaultValue = 4;
    desc.isQuantized = true;
    desc.quantizeStep = 1;
    list.push_back(desc);

    return list;
}

float SongPartitioner::getParameter(std::string name) const
{
    if (name == "bpb") return m_bpb;
    return 0.0;
}

void SongPartitioner::setParameter(std::string name, float value)
{
    if (name == "bpb") m_bpb = lrintf(value);
}


// Return the StepSize for Chroma Extractor 
size_t SongPartitioner::getPreferredStepSize() const
{
    size_t step = size_t(m_inputSampleRate * m_stepSecs + 0.0001);
    if (step < 1) step = 1;

    return step;
}

// Return the BlockSize for Chroma Extractor 
size_t SongPartitioner::getPreferredBlockSize() const
{
    size_t theoretical = getPreferredStepSize() * 2;
    theoretical *= m_chromaFramesizeFactor; 

    return theoretical;
}


// Initialize the plugin and define Beat Tracker and Chroma Extractor Objects
bool SongPartitioner::initialise(size_t channels, size_t stepSize, size_t blockSize)
{
    if (m_d) {
	delete m_d;
	m_d = 0;
    }

    if (channels < getMinChannelCount() ||
	channels > getMaxChannelCount()) {
        std::cerr << "SongPartitioner::initialise: Unsupported channel count: "
                  << channels << std::endl;
        return false;
    }

    if (stepSize != getPreferredStepSize()) {
        std::cerr << "ERROR: SongPartitioner::initialise: Unsupported step size for this sample rate: "
                  << stepSize << " (wanted " << (getPreferredStepSize()) << ")" << std::endl;
        return false;
    }

    if (blockSize != getPreferredBlockSize()) {
        std::cerr << "WARNING: SongPartitioner::initialise: Sub-optimal block size for this sample rate: "
                  << blockSize << " (wanted " << getPreferredBlockSize() << ")" << std::endl;
    }

    // Beat tracker and Chroma extractor has two different configuration parameters 
    
    // Configuration Parameters for Beat Tracker
    DFConfig dfConfig;
    dfConfig.DFType = DF_COMPLEXSD;
    dfConfig.stepSize = stepSize;
    dfConfig.frameLength = blockSize / m_chromaFramesizeFactor;
    dfConfig.dbRise = 3;
    dfConfig.adaptiveWhitening = false;
    dfConfig.whiteningRelaxCoeff = -1;
    dfConfig.whiteningFloor = -1;
    
    // Initialise Beat Tracker
    m_d = new BeatTrackerData(m_inputSampleRate, dfConfig);
    m_d->downBeat->setBeatsPerBar(m_bpb);
    
    // Initialise Chroma Extractor
    m_chromadata = new ChromaData(m_inputSampleRate, blockSize);
    m_chromadata->initialise();
    
    return true;
}

void SongPartitioner::reset()
{
    if (m_d) m_d->reset();
    m_pluginFrameCount = 0;
}

SongPartitioner::OutputList SongPartitioner::getOutputDescriptors() const
{
    OutputList list;
    size_t outputCounter = 0;

    OutputDescriptor beat;
    beat.identifier = "beats";
    beat.name = "Beats";
    beat.description = "Beat locations labelled with metrical position";
    beat.unit = "";
    beat.hasFixedBinCount = true;
    beat.binCount = 0;
    beat.sampleType = OutputDescriptor::VariableSampleRate;
    beat.sampleRate = 1.0 / m_stepSecs;
    m_beatOutputNumber = outputCounter++;

    OutputDescriptor bars;
    bars.identifier = "bars";
    bars.name = "Bars";
    bars.description = "Bar locations";
    bars.unit = "";
    bars.hasFixedBinCount = true;
    bars.binCount = 0;
    bars.sampleType = OutputDescriptor::VariableSampleRate;
    bars.sampleRate = 1.0 / m_stepSecs;
    m_barsOutputNumber = outputCounter++;

    OutputDescriptor beatcounts;
    beatcounts.identifier = "beatcounts";
    beatcounts.name = "Beat Count";
    beatcounts.description = "Beat counter function";
    beatcounts.unit = "";
    beatcounts.hasFixedBinCount = true;
    beatcounts.binCount = 1;
    beatcounts.sampleType = OutputDescriptor::VariableSampleRate;
    beatcounts.sampleRate = 1.0 / m_stepSecs;
    m_beatcountsOutputNumber = outputCounter++;

    OutputDescriptor beatsd;
    beatsd.identifier = "beatsd";
    beatsd.name = "Beat Spectral Difference";
    beatsd.description = "Beat spectral difference function used for bar-line detection";
    beatsd.unit = "";
    beatsd.hasFixedBinCount = true;
    beatsd.binCount = 1;
    beatsd.sampleType = OutputDescriptor::VariableSampleRate;
    beatsd.sampleRate = 1.0 / m_stepSecs;
    m_beatsdOutputNumber = outputCounter++;
    
    OutputDescriptor logscalespec;
    logscalespec.identifier = "logscalespec";
    logscalespec.name = "Log-Frequency Spectrum";
    logscalespec.description = "Spectrum with linear frequency on a log scale.";
    logscalespec.unit = "";
    logscalespec.hasFixedBinCount = true;
    logscalespec.binCount = nNote;
    logscalespec.hasKnownExtents = false;
    logscalespec.isQuantized = false;
    logscalespec.sampleType = OutputDescriptor::FixedSampleRate;
    logscalespec.hasDuration = false;
    logscalespec.sampleRate = m_inputSampleRate/2048;
    m_logscalespecOutputNumber = outputCounter++;
    
    OutputDescriptor bothchroma;
    bothchroma.identifier = "bothchroma";
    bothchroma.name = "Chromagram and Bass Chromagram";
    bothchroma.description = "Tuning-adjusted chromagram and bass chromagram (stacked on top of each other) from NNLS approximate transcription.";
    bothchroma.unit = "";
    bothchroma.hasFixedBinCount = true;
    bothchroma.binCount = 24;
    bothchroma.hasKnownExtents = false;
    bothchroma.isQuantized = false;
    bothchroma.sampleType = OutputDescriptor::FixedSampleRate;
    bothchroma.hasDuration = false;
    bothchroma.sampleRate = m_inputSampleRate/2048;
    m_bothchromaOutputNumber = outputCounter++;
    
    OutputDescriptor qchromafw;
    qchromafw.identifier = "qchromafw";
    qchromafw.name = "Pseudo-Quantised Chromagram and Bass Chromagram";
    qchromafw.description = "Pseudo-Quantised Chromagram and Bass Chromagram (frames between two beats are identical).";
    qchromafw.unit = "";
    qchromafw.hasFixedBinCount = true;
    qchromafw.binCount = 24;
    qchromafw.hasKnownExtents = false;
    qchromafw.isQuantized = false;
    qchromafw.sampleType = OutputDescriptor::FixedSampleRate;
    qchromafw.hasDuration = false;
    qchromafw.sampleRate = m_inputSampleRate/2048;
    m_qchromafwOutputNumber = outputCounter++;    
    
    OutputDescriptor qchroma;
    qchroma.identifier = "qchroma";
    qchroma.name = "Quantised Chromagram and Bass Chromagram";
    qchroma.description = "Quantised Chromagram and Bass Chromagram.";
    qchroma.unit = "";
    qchroma.hasFixedBinCount = true;
    qchroma.binCount = 24;
    qchroma.hasKnownExtents = false;
    qchroma.isQuantized = false;
    qchroma.sampleType = OutputDescriptor::FixedSampleRate;
    qchroma.hasDuration = true;
    m_qchromaOutputNumber = outputCounter++;

    OutputDescriptor segm;
    segm.identifier = "segm";
    segm.name = "Segmentation";
    segm.description = "Segmentation";
    segm.unit = "segment-type";
    segm.hasFixedBinCount = true;
    //segm.binCount = 24;
    segm.binCount = 1;
    segm.hasKnownExtents = true;
    segm.minValue = 1;
    segm.maxValue = 5;
    segm.isQuantized = true;
    segm.quantizeStep = 1;
    segm.sampleType = OutputDescriptor::VariableSampleRate;
    segm.hasDuration = true;
    m_segmOutputNumber = outputCounter++;
    
    
   /* 
    OutputList list;
    OutputDescriptor segmentation;
    segmentation.identifier = "segmentation";
    segmentation.name = "Segmentation";
    segmentation.description = "Segmentation";
    segmentation.unit = "segment-type";
    segmentation.hasFixedBinCount = true;
    segmentation.binCount = 1;
    segmentation.hasKnownExtents = true;
    segmentation.minValue = 1;
    segmentation.maxValue = nSegmentTypes;
    segmentation.isQuantized = true;
    segmentation.quantizeStep = 1;
    segmentation.sampleType = OutputDescriptor::VariableSampleRate;
    segmentation.sampleRate = m_inputSampleRate / getPreferredStepSize();
    list.push_back(segmentation);
    return list;
    */
    
    
    list.push_back(beat);
    list.push_back(bars);
    list.push_back(beatcounts);
    list.push_back(beatsd);
    list.push_back(logscalespec);
    list.push_back(bothchroma);
    list.push_back(qchromafw);
    list.push_back(qchroma);
    list.push_back(segm);

    return list;
}

// Executed for each frame - called from the host  

// We use time domain input, because DownBeat requires it -- so we
// use the time-domain version of DetectionFunction::process which
// does its own FFT.  It requires doubles as input, so we need to
// make a temporary copy

// We only support a single input channel
SongPartitioner::FeatureSet SongPartitioner::process(const float *const *inputBuffers,Vamp::RealTime timestamp)
{
    if (!m_d) {
	cerr << "ERROR: SongPartitioner::process: "
	     << "SongPartitioner has not been initialised"
	     << endl;
	return FeatureSet();
    }

    const int fl = m_d->dfConfig.frameLength;
#ifndef __GNUC__
    double *dfinput = (double *)alloca(fl * sizeof(double));
#else
    double dfinput[fl];
#endif
    int sampleOffset = ((m_chromaFramesizeFactor-1) * fl) / 2;
    
    // Since chroma needs a much longer frame size, we only ever use the very
    // beginning of the frame for beat tracking.
    for (int i = 0; i < fl; ++i) dfinput[i] = inputBuffers[0][i];
    double output = m_d->df->process(dfinput);

    if (m_d->dfOutput.empty()) m_d->origin = timestamp;

//    std::cerr << "df[" << m_d->dfOutput.size() << "] is " << output << std::endl;
    m_d->dfOutput.push_back(output);

    // Downsample and store the incoming audio block.
    // We have an overlap on the incoming audio stream (step size is
    // half block size) -- this function is configured to take only a
    // step size's worth, so effectively ignoring the overlap.  Note
    // however that this means we omit the last blocksize - stepsize
    // samples completely for the purposes of barline detection
    // (hopefully not a problem)
    m_d->downBeat->pushAudioBlock(inputBuffers[0]);

    // The following is not done every time, but only every m_chromaFramesizeFactor times,
    // because the chroma does not need dense time frames.
    
    if (m_pluginFrameCount % m_chromaStepsizeFactor == 0)
    {    
        
        // Window the full time domain, data, FFT it and process chroma stuff.
    
        #ifndef __GNUC__
            float *windowedBuffers = (float *)alloca(m_chromadata->blockSize * sizeof(float));
        #else
            float windowedBuffers[m_chromadata->blockSize];
        #endif
        m_chromadata->window.cut(&inputBuffers[0][0], &windowedBuffers[0]);
    
        // adjust timestamp (we want the middle of the frame)
        timestamp = timestamp + Vamp::RealTime::frame2RealTime(sampleOffset, lrintf(m_inputSampleRate));

        m_chromadata->baseProcess(&windowedBuffers[0], timestamp);
        
    }
    m_pluginFrameCount++;
    
    FeatureSet fs;
    fs[m_logscalespecOutputNumber].push_back(
        m_chromadata->logSpectrum.back());
    return fs;
}

SongPartitioner::FeatureSet SongPartitioner::getRemainingFeatures()
{
    if (!m_d) {
	cerr << "ERROR: SongPartitioner::getRemainingFeatures: "
	     << "SongPartitioner has not been initialised"
	     << endl;
	return FeatureSet();
    }

    FeatureSet masterFeatureset = BeatTrack();
    FeatureList chromaList = ChromaFeatures();
    
    for (size_t i = 0; i < chromaList.size(); ++i)
    {
        masterFeatureset[m_bothchromaOutputNumber].push_back(chromaList[i]);
    }
    
    // quantised and pseudo-quantised (beat-wise) chroma
    std::vector<FeatureList> quantisedChroma = BeatQuantiser(chromaList, masterFeatureset[m_beatOutputNumber]);
    
    masterFeatureset[m_qchromafwOutputNumber] = quantisedChroma[0];
    masterFeatureset[m_qchromaOutputNumber] = quantisedChroma[1];
    
    // Segmentation
    masterFeatureset[m_segmOutputNumber] = Segmenter(quantisedChroma[1]);
    
    return(masterFeatureset);
}

/* ------ Beat Tracker ------ */

SongPartitioner::FeatureSet SongPartitioner::BeatTrack()
{
    vector<double> df;
    vector<double> beatPeriod;
    vector<double> tempi;
    
    for (size_t i = 2; i < m_d->dfOutput.size(); ++i) { // discard first two elts
        df.push_back(m_d->dfOutput[i]);
        beatPeriod.push_back(0.0);
    }
    if (df.empty()) return FeatureSet();

    TempoTrackV2 tt(m_inputSampleRate, m_d->dfConfig.stepSize);
    tt.calculateBeatPeriod(df, beatPeriod, tempi);

    vector<double> beats;
    tt.calculateBeats(df, beatPeriod, beats);

    vector<int> downbeats;
    size_t downLength = 0;
    const float *downsampled = m_d->downBeat->getBufferedAudio(downLength);
    m_d->downBeat->findDownBeats(downsampled, downLength, beats, downbeats);

    vector<double> beatsd;
    m_d->downBeat->getBeatSD(beatsd);
    
    /*std::cout << "BeatTracker: found downbeats at: ";
    for (int i = 0; i < downbeats.size(); ++i) std::cout << downbeats[i] << " " << std::endl;*/
    
    FeatureSet returnFeatures;

    char label[20];

    int dbi = 0;
    int beat = 0;
    int bar = 0;

    if (!downbeats.empty()) {
        // get the right number for the first beat; this will be
        // incremented before use (at top of the following loop)
        int firstDown = downbeats[0];
        beat = m_bpb - firstDown - 1;
        if (beat == m_bpb) beat = 0;
    }

    for (size_t i = 0; i < beats.size(); ++i) {
        
        size_t frame = beats[i] * m_d->dfConfig.stepSize;
        
        if (dbi < downbeats.size() && i == downbeats[dbi]) {
            beat = 0;
            ++bar;
            ++dbi;
        } else {
            ++beat;
        }
        
        /* Ooutput Section */
        
        // outputs are:
        //
        // 0 -> beats
        // 1 -> bars
        // 2 -> beat counter function
        
        Feature feature;
        feature.hasTimestamp = true;
        feature.timestamp = m_d->origin + Vamp::RealTime::frame2RealTime (frame, lrintf(m_inputSampleRate));
        
        sprintf(label, "%d", beat + 1);
        feature.label = label;
        returnFeatures[m_beatOutputNumber].push_back(feature);          // labelled beats
        
        feature.values.push_back(beat + 1);
        returnFeatures[m_beatcountsOutputNumber].push_back(feature);    // beat function
        
        if (i > 0 && i <= beatsd.size()) {
            feature.values.clear();
            feature.values.push_back(beatsd[i-1]);
            feature.label = "";
            returnFeatures[m_beatsdOutputNumber].push_back(feature);    // beat spectral difference
        }
        
        if (beat == 0) {
            feature.values.clear();
            sprintf(label, "%d", bar);
            feature.label = label;
            returnFeatures[m_barsOutputNumber].push_back(feature);      // bars
        }
    }

    return returnFeatures;
}


/* ------ Chroma Extractor ------ */

SongPartitioner::FeatureList SongPartitioner::ChromaFeatures()
{
        
    FeatureList returnFeatureList;
    FeatureList tunedlogfreqspec;
    
    if (m_chromadata->logSpectrum.size() == 0) return returnFeatureList;

    /**  Calculate Tuning
         calculate tuning from (using the angle of the complex number defined by the 
         cumulative mean real and imag values)
    **/
    float meanTuningImag = 0;
    float meanTuningReal = 0;
    for (int iBPS = 0; iBPS < nBPS; ++iBPS) {
        meanTuningReal += m_chromadata->meanTunings[iBPS] * m_chromadata->cosvalues[iBPS];
        meanTuningImag += m_chromadata->meanTunings[iBPS] * m_chromadata->sinvalues[iBPS];
    }
    float cumulativetuning = 440 * pow(2,atan2(meanTuningImag, meanTuningReal)/(24*M_PI));
    float normalisedtuning = atan2(meanTuningImag, meanTuningReal)/(2*M_PI);
    int intShift = floor(normalisedtuning * 3);
    float floatShift = normalisedtuning * 3 - intShift; // floatShift is a really bad name for this
     
    char buffer0 [50];
 
    sprintf(buffer0, "estimated tuning: %0.1f Hz", cumulativetuning);
                 
    /** Tune Log-Frequency Spectrogram
        calculate a tuned log-frequency spectrogram (f2): use the tuning estimated above (kinda f0) to 
        perform linear interpolation on the existing log-frequency spectrogram (kinda f1).
    **/
    cerr << endl << "[NNLS Chroma Plugin] Tuning Log-Frequency Spectrogram ... ";
             
    float tempValue = 0;

    int count = 0;
    
    for (FeatureList::iterator i = m_chromadata->logSpectrum.begin(); i != m_chromadata->logSpectrum.end(); ++i) 
    {
        
        Feature f1 = *i;
        Feature f2; // tuned log-frequency spectrum
        
        f2.hasTimestamp = true;
        f2.timestamp = f1.timestamp;
        
        f2.values.push_back(0.0); 
        f2.values.push_back(0.0); // set lower edge to zero

        if (m_chromadata->tuneLocal) {
            intShift = floor(m_chromadata->localTuning[count] * 3);
            floatShift = m_chromadata->localTuning[count] * 3 - intShift; 
            // floatShift is a really bad name for this
        }

        for (int k = 2; k < (int)f1.values.size() - 3; ++k) 
        { // interpolate all inner bins
            tempValue = f1.values[k + intShift] * (1-floatShift) + f1.values[k+intShift+1] * floatShift;
            f2.values.push_back(tempValue);
        }
         
        f2.values.push_back(0.0); 
        f2.values.push_back(0.0); 
        f2.values.push_back(0.0); // upper edge

        vector<float> runningmean = SpecialConvolution(f2.values,m_chromadata->hw);
        vector<float> runningstd;
        for (int i = 0; i < nNote; i++) { // first step: squared values into vector (variance)
            runningstd.push_back((f2.values[i] - runningmean[i]) * (f2.values[i] - runningmean[i]));
        }
        runningstd = SpecialConvolution(runningstd,m_chromadata->hw); // second step convolve
        for (int i = 0; i < nNote; i++) 
        { 
            
            runningstd[i] = sqrt(runningstd[i]); 
            // square root to finally have running std
            
            if (runningstd[i] > 0) 
            {
                f2.values[i] = (f2.values[i] - runningmean[i]) > 0 ?
                    (f2.values[i] - runningmean[i]) / pow(runningstd[i],m_chromadata->whitening) : 0;
            }
            
            if (f2.values[i] < 0) {
                
                cerr << "ERROR: negative value in logfreq spectrum" << endl;
                
            }
        }
        tunedlogfreqspec.push_back(f2);
        count++;
    }
    cerr << "done." << endl;    
    /** Semitone spectrum and chromagrams
        Semitone-spaced log-frequency spectrum derived 
        from the tuned log-freq spectrum above. the spectrum
        is inferred using a non-negative least squares algorithm.
        Three different kinds of chromagram are calculated, "treble", "bass", and "both" (which means 
        bass and treble stacked onto each other).
    **/
    if (m_chromadata->useNNLS == 0) {
        cerr << "[NNLS Chroma Plugin] Mapping to semitone spectrum and chroma ... ";
    } else {
        cerr << "[NNLS Chroma Plugin] Performing NNLS and mapping to chroma ... ";
    }
    
    vector<float> oldchroma = vector<float>(12,0);
    vector<float> oldbasschroma = vector<float>(12,0);
    count = 0;

    for (FeatureList::iterator it = tunedlogfreqspec.begin(); it != tunedlogfreqspec.end(); ++it) {
        Feature logfreqsp = *it; // logfreq spectrum
        Feature bothchroma; // treble and bass chromagram
                            
        bothchroma.hasTimestamp = true;
        bothchroma.timestamp = logfreqsp.timestamp;
        
        float b[nNote];

        bool some_b_greater_zero = false;
        float sumb = 0;
        for (int i = 0; i < nNote; i++) {
            b[i] = logfreqsp.values[i];
            sumb += b[i];
            if (b[i] > 0) {
                some_b_greater_zero = true;
            }            
        }
    
        // here's where the non-negative least squares algorithm calculates the note activation x

        vector<float> chroma = vector<float>(12, 0);
        vector<float> basschroma = vector<float>(12, 0);
        float currval;
        int iSemitone = 0;
     
        if (some_b_greater_zero) {
            if (m_chromadata->useNNLS == 0) {
                for (int iNote = nBPS/2 + 2; iNote < nNote - nBPS/2; iNote += nBPS) {
                    currval = 0;
                    for (int iBPS = -nBPS/2; iBPS < nBPS/2+1; ++iBPS) {
                        currval += b[iNote + iBPS] * (1-abs(iBPS*1.0/(nBPS/2+1)));                       
                    }
                    chroma[iSemitone % 12] += currval * treblewindow[iSemitone];
                    basschroma[iSemitone % 12] += currval * basswindow[iSemitone];
                    iSemitone++;
                }
         
            } else {
                float x[84+1000];
                for (int i = 1; i < 1084; ++i) x[i] = 1.0;
                vector<int> signifIndex;
                int index=0;
                sumb /= 84.0;
                for (int iNote = nBPS/2 + 2; iNote < nNote - nBPS/2; iNote += nBPS) {
                    float currval = 0;
                    for (int iBPS = -nBPS/2; iBPS < nBPS/2+1; ++iBPS) {
                        currval += b[iNote + iBPS]; 
                    }
                    if (currval > 0) signifIndex.push_back(index);
                    index++;
                }
                float rnorm;
                float w[84+1000];
                float zz[84+1000];
                int indx[84+1000];
                int mode;
                int dictsize = nNote*signifIndex.size();

                float *curr_dict = new float[dictsize];
                for (int iNote = 0; iNote < (int)signifIndex.size(); ++iNote) {
                    for (int iBin = 0; iBin < nNote; iBin++) {
                        curr_dict[iNote * nNote + iBin] = 
                            1.0 * m_chromadata->dict[signifIndex[iNote] * nNote + iBin];
                    }
                }
                nnls(curr_dict, nNote, nNote, signifIndex.size(), b, x, &rnorm, w, zz, indx, &mode);
                delete [] curr_dict;
                for (int iNote = 0; iNote < (int)signifIndex.size(); ++iNote) {
                    // cerr << mode << endl;
                    chroma[signifIndex[iNote] % 12] += x[iNote] * treblewindow[signifIndex[iNote]];
                    basschroma[signifIndex[iNote] % 12] += x[iNote] * basswindow[signifIndex[iNote]];
                }
            }    
        }
 
        chroma.insert(chroma.begin(), basschroma.begin(), basschroma.end()); 
        // just stack the both chromas 
        
        bothchroma.values = chroma; 
        returnFeatureList.push_back(bothchroma);
        count++;
    }
    cerr << "done." << endl;

    return returnFeatureList;     
}

/* ------ Beat Quantizer ------ */

std::vector<Vamp::Plugin::FeatureList>
SongPartitioner::BeatQuantiser(Vamp::Plugin::FeatureList chromagram, Vamp::Plugin::FeatureList beats)
{
    std::vector<FeatureList> returnVector;
    
    FeatureList fwQchromagram; // frame-wise beat-quantised chroma
    FeatureList bwQchromagram; // beat-wise beat-quantised chroma
    
    int nChromaFrame = (int) chromagram.size();
    int nBeat = (int) beats.size();
    
    if (nBeat == 0 && nChromaFrame == 0) return returnVector;
    
    size_t nBin = chromagram[0].values.size();
    
    vector<float> tempChroma = vector<float>(nBin);
    
    Vamp::RealTime beatTimestamp = Vamp::RealTime::zeroTime;
    int currBeatCount = -1; // start before first beat
    int framesInBeat = 0;
    
    for (int iChroma = 0; iChroma < nChromaFrame; ++iChroma)
    {
        Vamp::RealTime frameTimestamp = chromagram[iChroma].timestamp;
		Vamp::RealTime tempBeatTimestamp;
		
		if (currBeatCount != beats.size()-1) tempBeatTimestamp = beats[currBeatCount+1].timestamp;
		else tempBeatTimestamp = chromagram[nChromaFrame-1].timestamp;
		
        if (frameTimestamp > tempBeatTimestamp ||
            iChroma == nChromaFrame-1)
        {
            // new beat (or last chroma frame)
            // 1. finish all the old beat processing
			if (framesInBeat > 0)
			{
            	for (int i = 0; i < nBin; ++i) tempChroma[i] /= framesInBeat; // average
			}
            
            Feature bwQchromaFrame;
            bwQchromaFrame.hasTimestamp = true;
            bwQchromaFrame.timestamp = beatTimestamp;
            bwQchromaFrame.values = tempChroma;
            bwQchromaFrame.duration = beats[currBeatCount+1].timestamp - beats[currBeatCount].timestamp;
            bwQchromagram.push_back(bwQchromaFrame);
            
            for (int iFrame = -framesInBeat; iFrame < 0; ++iFrame)
            {
                Feature fwQchromaFrame;
                fwQchromaFrame.hasTimestamp = true;
                fwQchromaFrame.timestamp = chromagram[iChroma+iFrame].timestamp;
                fwQchromaFrame.values = tempChroma; // all between two beats get the same
                fwQchromagram.push_back(fwQchromaFrame);
            }
            
            // 2. increments / resets for current (new) beat
            currBeatCount++;
            beatTimestamp = beats[currBeatCount].timestamp;
            for (size_t i = 0; i < nBin; ++i) tempChroma[i] = 0; // average
            framesInBeat = 0;
        }
        framesInBeat++;
        for (size_t i = 0; i < nBin; ++i) tempChroma[i] += chromagram[iChroma].values[i];
    }
    returnVector.push_back(fwQchromagram);
    returnVector.push_back(bwQchromagram);
}

/* -------------------------------- */
/* ------ Support Functions  ------ */
/* -------------------------------- */

// one-dimesion median filter
arma::vec medfilt1(arma::vec v, int medfilt_length)
{    
    int halfWin = medfilt_length/2;
    
    // result vector
    arma::vec res = arma::zeros<arma::vec>(v.size());
    
    // padding 
    arma::vec padV = arma::zeros<arma::vec>(v.size()+medfilt_length-1);
    
    for (unsigned i=medfilt_length/2; i < medfilt_length/2+v.size(); ++ i)
    {
        padV(i) = v(i-medfilt_length/2);
    }    
    
    // Median filter
    arma::vec win = arma::zeros<arma::vec>(medfilt_length);
    
    for (unsigned i=0; i < v.size(); ++i)
    {
        win = padV.subvec(i,i+halfWin*2);
        win = sort(win);
        res(i) = win(halfWin);
    }
    
    return res;
}


// Quantile
double quantile(arma::vec v, double p)
{
    arma::vec sortV = arma::sort(v);
    int n = sortV.size();
    arma::vec x = arma::zeros<vec>(n+2);
    arma::vec y = arma::zeros<vec>(n+2);
    
    x(0) = 0;
    x(n+1) = 100; 
    
    for (unsigned i=1; i<n+1; ++i)
        x(i) = 100*(0.5+(i-1))/n;
        
    y(0) = sortV(0);
    y.subvec(1,n) = sortV;
    y(n+1) = sortV(n-1);
    
    arma::uvec x2index = find(x>=p*100);
    
    // Interpolation
    double x1 = x(x2index(0)-1);
    double x2 = x(x2index(0));
    double y1 = y(x2index(0)-1);
    double y2 = y(x2index(0));
    
    double res = (y2-y1)/(x2-x1)*(p*100-x1)+y1;
    
    return res;
}

// Max Filtering
arma::mat maxfilt1(arma::mat inmat, int len)
{
    arma::mat outmat = inmat;
    
    for (int i=0; i<inmat.n_rows; ++i)
    {
        if (arma::sum(inmat.row(i)) > 0)
        {
            // Take a window of rows
            int startWin;
            int endWin;
            
            if (0 > i-len)
                startWin = 0;
            else
                startWin = i-len;
            
            if (inmat.n_rows-1 < i+len-1)
                endWin = inmat.n_rows-1;
            else
                endWin = i+len-1;
    
            outmat(i,span::all) = arma::max(inmat(span(startWin,endWin),span::all));
        }
    }
    
    return outmat;
    
}

// Null Parts
Part nullpart(vector<Part> parts, arma::vec barline)
{
    arma::uvec nullindices = arma::ones<arma::uvec>(barline.size());
    for (unsigned iPart=0; iPart<parts.size(); ++iPart)
    {
        for (unsigned iIndex=0; iIndex<parts[0].indices.size(); ++iIndex) 
        {
            for (unsigned i=0; i<parts[iPart].n; ++i) 
            {
                unsigned ind = parts[iPart].indices[iIndex]+i;
                nullindices(ind) = 0;
            }
        }
    }
    
    Part newPart;
    newPart.n = 1;
    uvec q = find(nullindices > 0);
    
    for (unsigned i=0; i<q.size();++i) 
        newPart.indices.push_back(q(i));
        
    newPart.letter = '-';
    newPart.value = 0;
    newPart.level = 0;
    
    return newPart;    
}


// Merge Nulls
void mergenulls(vector<Part> &parts)
{
    for (unsigned iPart=0; iPart<parts.size(); ++iPart)
    {
        
        vector<Part> newVectorPart;
        
        if (parts[iPart].letter.compare("-")==0)
        {
            sort (parts[iPart].indices.begin(), parts[iPart].indices.end());
            unsigned newpartind = -1;
            
            vector<int> indices;
            indices.push_back(-2);
            
            for (unsigned iIndex=0; iIndex<parts[iPart].indices.size(); ++iIndex) 
                indices.push_back(parts[iPart].indices[iIndex]);
            
            for (unsigned iInd=1; iInd < indices.size(); ++iInd)
            { 
                if (indices[iInd] - indices[iInd-1] > 1)
                {
                    newpartind++;

                    Part newPart;
                    newPart.letter = 'n';
                    std::stringstream out;
                    out << newpartind+1;
                    newPart.letter.append(out.str());
                    newPart.value = 20+newpartind+1;
                    newPart.n = 1;
                    newPart.indices.push_back(indices[iInd]);
                    newPart.level = 0;   
                    
                    newVectorPart.push_back(newPart);
                }
                else
                {
                    newVectorPart[newpartind].n = newVectorPart[newpartind].n+1;
                }
            }
            parts.erase (parts.end());
            
            for (unsigned i=0; i<newVectorPart.size(); ++i)
                parts.push_back(newVectorPart[i]);
        }
    }
}

/* ------ Segmentation ------ */

vector<Part> songSegment(Vamp::Plugin::FeatureList quatisedChromagram)
{
    
    
    /* ------ Parameters ------ */
    double thresh_beat = 0.85;
    double thresh_seg = 0.80;
    int medfilt_length = 5;
    int minlength = 28;
    int maxlength = 128;
    double quantilePerc = 0.1;
    /* ------------------------ */
    
    
    // Collect Info
    int nBeat = quatisedChromagram.size();                      // Number of feature vector
    int nFeatValues = quatisedChromagram[0].values.size();      // Number of values for each feature vector
    
    arma::irowvec timeStamp = arma::zeros<arma::imat>(1,nBeat);       // Vector of Time Stamps
    
	// Save time stamp as a Vector
    if (quatisedChromagram[0].hasTimestamp)
    {
        for (unsigned i = 0; i < nBeat; ++ i)
            timeStamp[i] = quatisedChromagram[i].timestamp.nsec;
    }
    
    
    // Build a ObservationTOFeatures Matrix
    arma::mat featVal = arma::zeros<mat>(nBeat,nFeatValues/2);
    
    for (unsigned i = 0; i < nBeat; ++ i)
        for (unsigned j = 0; j < nFeatValues/2; ++ j)
        {
            featVal(i,j) = (quatisedChromagram[i].values[j]+quatisedChromagram[i].values[j+12]) * 0.8;
        }
    
    // Set to arbitrary value to feature vectors with low std
    arma::mat a = stddev(featVal,1,1);
    
    // Feature Colleration Matrix
    arma::mat simmat0 = 1-arma::cor(arma::trans(featVal));
    

    for (unsigned i = 0; i < nBeat; ++ i)
    {
        if (a(i)<0.000001)
        {
            featVal(i,1) = 1000;  // arbitrary  
            
            for (unsigned j = 0; j < nFeatValues/2; ++j)
            {
                simmat0(i,j) = 1;
                simmat0(j,i) = 1;
            }
        }
    }
    
    arma::mat simmat = 1-simmat0/2;
    
    // -------- To delate when the proble with the add of beat will be solved -------
    for (unsigned i = 0; i < nBeat; ++ i)
     for (unsigned j = 0; j < nBeat; ++ j)
         if (!std::isfinite(simmat(i,j)))
             simmat(i,j)=0;
    // ------------------------------------------------------------------------------
    
    // Median Filtering applied to the Correlation Matrix
    // The median filter is for each diagonal of the Matrix
    arma::mat median_simmat = arma::zeros<arma::mat>(nBeat,nBeat);
    
    for (unsigned i = 0; i < nBeat; ++ i)
    {
        arma::vec temp = medfilt1(simmat.diag(i),medfilt_length);
        median_simmat.diag(i) = temp;
        median_simmat.diag(-i) = temp;
    }

    for (unsigned i = 0; i < nBeat; ++ i)
        for (unsigned j = 0; j < nBeat; ++ j)
            if (!std::isfinite(median_simmat(i,j)))
                median_simmat(i,j) = 0;
    
    // -------------- NOT CONVERTED -------------------------------------    
    //    if param.seg.standardise
    //        med_median_simmat = repmat(median(median_simmat),nBeat,1);
    //    std_median_simmat = repmat(std(median_simmat),nBeat,1);
    //    median_simmat = (median_simmat - med_median_simmat) ./ std_median_simmat;
    //    end
    // --------------------------------------------------------
    
    // Retrieve Bar Bounderies
    arma::uvec dup = find(median_simmat > thresh_beat);
    arma::mat potential_duplicates = arma::zeros<arma::mat>(nBeat,nBeat);
    potential_duplicates.elem(dup) = arma::ones<arma::vec>(dup.size());
    potential_duplicates = trimatu(potential_duplicates);
    
    unsigned nPartlengths = round((maxlength-minlength)/4)+1;
    arma::vec partlengths = zeros<arma::vec>(nPartlengths);
    
    for (unsigned i = 0; i < nPartlengths; ++ i)
        partlengths(i) = (i*4)+ minlength;
    
    // initialise arrays
    arma::cube simArray = zeros<arma::cube>(nBeat,nBeat,nPartlengths);
    arma::cube decisionArray2 = zeros<arma::cube>(nBeat,nBeat,nPartlengths);

    int conta = 0;
       
    //for (unsigned iLength = 0; iLength < nPartlengths; ++ iLength)
    for (unsigned iLength = 0; iLength < 20; ++ iLength)
    {
        unsigned len = partlengths(iLength);
        unsigned nUsedBeat = nBeat - len + 1;                   // number of potential rep beginnings: they can't overlap at the end of the song
        
        for (unsigned iBeat = 0; iBeat < nUsedBeat; ++ iBeat)   // looping over all columns (arbitrarily chosen columns)
        {
            arma::uvec help2 = find(potential_duplicates(span(0,nUsedBeat-1),iBeat)==1);
            
            for (unsigned i=0; i<help2.size(); ++i)
            {

                // measure how well two length len segments go together
                int kBeat = help2(i);
                arma::vec distrib = median_simmat(span(iBeat,iBeat+len-1),span(kBeat,kBeat+len-1)).diag(0);
                simArray(iBeat,kBeat,iLength) = quantile(distrib,quantilePerc);
            }
        }
        
        arma::mat tempM = simArray(span(0,nUsedBeat-1),span(0,nUsedBeat-1),span(iLength,iLength));
        simArray.slice(iLength)(span(0,nUsedBeat-1),span(0,nUsedBeat-1)) = tempM + arma::trans(tempM) - (eye<mat>(nUsedBeat,nUsedBeat)%tempM); 
        
        // convolution
        arma::vec K = arma::zeros<vec>(3);
        K << 0.01 << 0.98 << 0.01;
        
        
        for (unsigned i=0; i<simArray.n_rows; ++i)
        {
            arma::rowvec t = arma::conv((arma::rowvec)simArray.slice(iLength).row(i),K);
            simArray.slice(iLength)(i,span::all) = t.subvec(1,t.size()-2);
        }
 
        // take only over-average bars that do not overlap
        
        arma::mat temp = arma::zeros<mat>(simArray.n_rows, simArray.n_cols);
        temp(span::all, span(0,nUsedBeat-1)) = simArray.slice(iLength)(span::all,span(0,nUsedBeat-1));
        
        for (unsigned i=0; i<temp.n_rows; ++i)
            for (unsigned j=0; j<nUsedBeat; ++j)
                if (temp(i,j) < thresh_seg)
                    temp(i,j) = 0;
        
        decisionArray2.slice(iLength) = temp;

        arma::mat maxMat = maxfilt1(decisionArray2.slice(iLength),len-1);
        
        for (unsigned i=0; i<decisionArray2.n_rows; ++i)
            for (unsigned j=0; j<decisionArray2.n_cols; ++j)
                if (decisionArray2.slice(iLength)(i,j) < maxMat(i,j))
                    decisionArray2.slice(iLength)(i,j) = 0;
        
        decisionArray2.slice(iLength) = decisionArray2.slice(iLength) % arma::trans(decisionArray2.slice(iLength));
        
        for (unsigned i=0; i<simArray.n_rows; ++i)
            for (unsigned j=0; j<simArray.n_cols; ++j)
                if (simArray.slice(iLength)(i,j) < thresh_seg)
                    potential_duplicates(i,j) = 0; 
    }
    
    // Milk the data
    
    arma::mat bestval;
    
    for (unsigned iLength=0; iLength<nPartlengths; ++iLength)
    {
        arma::mat temp = arma::zeros<arma::mat>(decisionArray2.n_rows,decisionArray2.n_cols);

       for (unsigned rows=0; rows<decisionArray2.n_rows; ++rows)
            for (unsigned cols=0; cols<decisionArray2.n_cols; ++cols)
                if (decisionArray2.slice(iLength)(rows,cols) > 0)
                    temp(rows,cols) = 1;
        
        arma::vec currLogicSum = arma::sum(temp,1);
        
        for (unsigned iBeat=0; iBeat<nBeat; ++iBeat)
            if (currLogicSum(iBeat) > 1)
            {
                arma::vec t = decisionArray2.slice(iLength)(span::all,iBeat);
                double currSum = sum(t);
                
                unsigned count = 0;
                for (unsigned i=0; i<t.size(); ++i)
                    if (t(i)>0)
                        count++;
                
                currSum = (currSum/count)/2;
                
                arma::rowvec t1;
                t1 << (currLogicSum(iBeat)-1) * partlengths(iLength) << currSum << iLength << iBeat << currLogicSum(iBeat);
                
                bestval = join_cols(bestval,t1);
            }
    }
    
    // Definition of the resulting vector
    vector<Part> parts;
    
    // make a table of all valid sets of parts
    
    char partletters[] = {'A','B','C','D','E','F','G', 'H','I','J','K','L','M','N','O','P','Q','R','S'};
    unsigned partvalues[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
    arma::vec valid_sets = arma::ones<arma::vec>(bestval.n_rows);
    
    if (!bestval.is_empty())
    {
        
        // In questo punto viene introdotto un errore alla 3 cifra decimale
        
        arma::colvec t = arma::zeros<arma::colvec>(bestval.n_rows);
        for (unsigned i=0; i<bestval.n_rows; ++i)
        {
            t(i) = bestval(i,1)*2;
        }
        
        double m = t.max();
        
        bestval(span::all,1) = bestval(span::all,1) / m; 
        bestval(span::all,0) = bestval(span::all,0) + bestval(span::all,1);
        
        arma::mat bestval2;
        for (unsigned i=0; i<bestval.n_cols; ++i)
            if (i!=1)
                bestval2 = join_rows(bestval2,bestval.col(i));
        
        for (unsigned kSeg=0; kSeg<6; ++kSeg)
        {
            arma::mat currbestvals = arma::zeros<arma::mat>(bestval2.n_rows, bestval2.n_cols);
            for (unsigned i=0; i<bestval2.n_rows; ++i)
                for (unsigned j=0; j<bestval2.n_cols; ++j)
                    if (valid_sets(i))
                        currbestvals(i,j) = bestval2(i,j);
            
            arma::vec t1 = currbestvals.col(0);
            double ma;
            uword maIdx;
            ma = t1.max(maIdx);
            
            std::cout << maIdx << " - " << ma << std::endl;
            
            if ((maIdx == 0)&&(ma == 0))
                break;

            double bestLength = partlengths(currbestvals(maIdx,1));
            arma::rowvec bestIndices = decisionArray2.slice(currbestvals(maIdx,1))(currbestvals(maIdx,2),span::all);
                
            arma::rowvec bestIndicesMap = arma::zeros<arma::rowvec>(bestIndices.size());
            for (unsigned i=0; i<bestIndices.size(); ++i)
                if (bestIndices(i)>0)
                    bestIndicesMap(i) = 1;
                   
            arma::rowvec mask = arma::zeros<arma::rowvec>(bestLength*2-1);
            for (unsigned i=0; i<bestLength; ++i)
                mask(i+bestLength-1) = 1;
            
            arma::rowvec t2 = arma::conv(bestIndicesMap,mask); 
            arma::rowvec island = t2.subvec(mask.size()/2,t2.size()-1-mask.size()/2);
            
            // Save results in the structure
            Part newPart;
            newPart.n = bestLength;
            uvec q1 = find(bestIndices > 0);
            
            for (unsigned i=0; i<q1.size();++i)
                newPart.indices.push_back(q1(i));
            
            newPart.letter = partletters[kSeg];
            newPart.value = partvalues[kSeg];
            newPart.level = kSeg+1;
            parts.push_back(newPart);
            
            uvec q2 = find(valid_sets==1);
            
            for (unsigned i=0; i<q2.size(); ++i)
            {
                unsigned iSet = q2(i);
                unsigned s = partlengths(bestval2(iSet,1));
                
                arma::rowvec mask1 = arma::zeros<arma::rowvec>(s*2-1);
                for (unsigned i=0; i<s; ++i)
                    mask1(i+s-1) = 1;
                
                arma::rowvec Ind = decisionArray2.slice(bestval2(iSet,1))(bestval2(iSet,2),span::all);
                arma::rowvec IndMap = arma::zeros<arma::rowvec>(Ind.size());
                for (unsigned i=0; i<Ind.size(); ++i)
                    if (Ind(i)>0)
                        IndMap(i) = 2;
                
                arma::rowvec t3 = arma::conv(IndMap,mask1); 
                arma::rowvec currislands = t3.subvec(mask1.size()/2,t3.size()-1-mask1.size()/2);       
                arma::rowvec islandsdMult = currislands%island;
                
                arma::uvec islandsIndex = find(islandsdMult > 0);
                
                if (islandsIndex.size() > 0)
                    valid_sets(iSet) = 0;
            }
        }
    }
    else
    {
        Part newPart;
        newPart.n = nBeat;
        newPart.indices.push_back(1);
        newPart.letter = 'A';
        newPart.value = 1;
        newPart.level = 1;
        parts.push_back(newPart);
    }
   
    arma::vec bar = linspace(1,nBeat,nBeat);    
    Part np = nullpart(parts,bar);
    parts.push_back(np);
    
    // -------------- NOT CONVERTED -------------------------------------  
    // if param.seg.editor
    //    [pa, ta] = partarray(parts);
    //    parts = editorssearch(pa, ta, parts);
    //    parts = [parts, nullpart(parts,1:nBeat)];
    // end
    // ------------------------------------------------------------------

    
    mergenulls(parts);
    
    
    // -------------- NOT CONVERTED -------------------------------------  
    // if param.seg.editor
    //    [pa, ta] = partarray(parts);
    //    parts = editorssearch(pa, ta, parts);
    //    parts = [parts, nullpart(parts,1:nBeat)];
    // end
    // ------------------------------------------------------------------
    
    return parts;
}



void songSegmentChroma(Vamp::Plugin::FeatureList quatisedChromagram, vector<Part> &parts)
{
    // Collect Info
    int nBeat = quatisedChromagram.size();                      // Number of feature vector
    int nFeatValues = quatisedChromagram[0].values.size();      // Number of values for each feature vector

    arma::mat synchTreble = arma::zeros<mat>(nBeat,nFeatValues/2);
    
    for (unsigned i = 0; i < nBeat; ++ i)
        for (unsigned j = 0; j < nFeatValues/2; ++ j)
        {
            synchTreble(i,j) = quatisedChromagram[i].values[j];
        }
    
    arma::mat synchBass = arma::zeros<mat>(nBeat,nFeatValues/2);
    
    for (unsigned i = 0; i < nBeat; ++ i)
        for (unsigned j = 0; j < nFeatValues/2; ++ j)
        {
            synchBass(i,j) = quatisedChromagram[i].values[j+12];
        }

    // Process
    
    arma::mat segTreble = arma::zeros<arma::mat>(quatisedChromagram.size(),quatisedChromagram[0].values.size()/2);
    arma::mat segBass = arma::zeros<arma::mat>(quatisedChromagram.size(),quatisedChromagram[0].values.size()/2);
    
    for (unsigned iPart=0; iPart<parts.size(); ++iPart)
    {
        parts[iPart].nInd = parts[iPart].indices.size();
        
        for (unsigned kOccur=0; kOccur<parts[iPart].nInd; ++kOccur)
        {
            int kStartIndex = parts[iPart].indices[kOccur];
            int kEndIndex = kStartIndex + parts[iPart].n-1;
            
            segTreble.rows(kStartIndex,kEndIndex) = segTreble.rows(kStartIndex,kEndIndex) + synchTreble.rows(kStartIndex,kEndIndex);
            segBass.rows(kStartIndex,kEndIndex) = segBass.rows(kStartIndex,kEndIndex) + synchBass.rows(kStartIndex,kEndIndex);
        }
    }
}


// Segment Integration
vector<Part> songSegmentIntegration(vector<Part> &parts)
{
    // Break up parts (every part will have one instance)
    vector<Part> newPartVector;
    vector<int> partindices;
    
    for (unsigned iPart=0; iPart<parts.size(); ++iPart)
    {
        parts[iPart].nInd = parts[iPart].indices.size();
        for (unsigned iInstance=0; iInstance<parts[iPart].nInd; ++iInstance)
        {
            Part newPart;
            newPart.n = parts[iPart].n;
            newPart.letter = parts[iPart].letter;
            newPart.value = parts[iPart].value;
            newPart.level = parts[iPart].level;
            newPart.indices.push_back(parts[iPart].indices[iInstance]);
            newPart.nInd = 1;
            partindices.push_back(parts[iPart].indices[iInstance]);
            
            newPartVector.push_back(newPart);
        }
    }
    
    
    // Sort the parts in order of occurrence
    sort (partindices.begin(), partindices.end());
    
    for (unsigned i=0; i<partindices.size(); ++i)
    {
        bool found = false;
        int in=0;    
        while (!found)
        {
            if (newPartVector[in].indices[0] == partindices[i])
            {
                newPartVector.push_back(newPartVector[in]);
                newPartVector.erase(newPartVector.begin()+in);
                found = true;
            }
            else
                in++;
        }  
    }
    
    // Clear the vector
    for (unsigned iNewpart=1; iNewpart < newPartVector.size(); ++iNewpart)
    {
        if (newPartVector[iNewpart].n < 12)
        {
            newPartVector[iNewpart-1].n = newPartVector[iNewpart-1].n + newPartVector[iNewpart].n;
            newPartVector.erase(newPartVector.begin()+iNewpart);
        }
    }

    return newPartVector;
}

// Segmenter
Vamp::Plugin::FeatureList SongPartitioner::Segmenter(Vamp::Plugin::FeatureList quatisedChromagram)
{
    /* --- Display Information --- */
    int numBeat = quatisedChromagram.size();
    int numFeats = quatisedChromagram[0].values.size();

    vector<Part> parts;
    vector<Part> finalParts;
    
    parts = songSegment(quatisedChromagram);
        
    songSegmentChroma(quatisedChromagram,parts);
    finalParts = songSegmentIntegration(parts);
    
    
    // TEMP ----
    /*for (unsigned i=0;i<finalParts.size(); ++i)
     {
     std::cout << "Parts n° " << i << std::endl;
     std::cout << "n°: " << finalParts[i].n << std::endl;
     std::cout << "letter: " <<  finalParts[i].letter << std::endl;
     
     std::cout << "indices: ";
     for (unsigned j=0;j<finalParts[i].indices.size(); ++j)
         std::cout << finalParts[i].indices[j] << " ";
       
     std::cout << std::endl;
     std::cout <<  "level: " << finalParts[i].level << std::endl;
     }*/
    
    // ---------
    
    
    // Output

    Vamp::Plugin::FeatureList results;
    
    
    Feature seg;
    
    arma::vec indices;
    unsigned idx=0;
    vector<int> values;
    vector<string> letters;
    
    for (unsigned iPart=0; iPart<finalParts.size()-1; ++iPart)
    {
        unsigned iInstance=0;
        seg.hasTimestamp = true;
         
        int ind = finalParts[iPart].indices[iInstance];
        int ind1 = finalParts[iPart+1].indices[iInstance];
         
        seg.timestamp = quatisedChromagram[ind].timestamp;
        seg.hasDuration = true;
        seg.duration = quatisedChromagram[ind1].timestamp-quatisedChromagram[ind].timestamp;
        seg.values.clear();
        seg.values.push_back(finalParts[iPart].value);
        seg.label = finalParts[iPart].letter;
         
        results.push_back(seg);
    }
    
    int ind = finalParts[finalParts.size()-1].indices[0];
    seg.timestamp = quatisedChromagram[ind].timestamp;
    seg.hasDuration = true;
    seg.duration = quatisedChromagram[quatisedChromagram.size()-1].timestamp-quatisedChromagram[ind].timestamp;
    seg.values.clear();
    seg.values.push_back(finalParts[finalParts.size()-1].value);
    seg.label = finalParts[finalParts.size()-1].letter;
    
    results.push_back(seg);

    return results;    
}

















