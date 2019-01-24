/* -*- c-basic-offset: 4 indent-tabs-mode: nil -*-  vi:set ts=8 sts=4 sw=4: */

/*
    Segmentino

    Code by Massimiliano Zanoni and Matthias Mauch
    Centre for Digital Music, Queen Mary, University of London

    Copyright 2009-2013 Queen Mary, University of London.

    This program is free software: you can redistribute it and/or
    modify it under the terms of the GNU Affero General Public License
    as published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version. See the file
    COPYING included with this distribution for more information.
*/

#include "Segmentino.h"

#include <qm-dsp/base/Window.h>
#include <qm-dsp/dsp/onsets/DetectionFunction.h>
#include <qm-dsp/dsp/onsets/PeakPicking.h>
#include <qm-dsp/dsp/transforms/FFT.h>
#include <qm-dsp/dsp/tempotracking/TempoTrackV2.h>
#include <qm-dsp/dsp/tempotracking/DownBeat.h>
#include <qm-dsp/maths/MathUtilities.h>

#include <nnls-chroma/chromamethods.h>

#include <armadillo>

#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>

#include <vamp-sdk/Plugin.h>

using arma::colvec;
using arma::conv;
using arma::cor;
using arma::cube;
using arma::eye;
using arma::imat;
using arma::irowvec;
using arma::linspace;
using arma::mat;
using arma::max;
using arma::ones;
using arma::rowvec;
using arma::sort;
using arma::span;
using arma::sum;
using arma::trans;
using arma::uvec;
using arma::uword;
using arma::vec;
using arma::zeros;

using std::string;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;

// Result Struct
typedef struct Part {
    int n;
    vector<int> indices;
    string letter;
    int value;
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
    int blockSize;
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
    int inputSampleRate;
    
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
        for (int i = 0; i < blockSize; i++) inputBuffersDouble[i] = inputBuffers[i];
        
        fft.forward(inputBuffersDouble, fftReal, fftImag);
        
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

        return true;
    }    
};


/* --------------------------------- */
/* ----- SONG PARTITIONER ---------- */
/* --------------------------------- */


/* --- ATTRIBUTES --- */

float Segmentino::m_stepSecs = 0.01161;            // 512 samples at 44100
int Segmentino::m_chromaFramesizeFactor = 16;   // 16 times as long as beat tracker's
int Segmentino::m_chromaStepsizeFactor = 4;     // 4 times as long as beat tracker's


/* --- METHODS --- */

/* --- Constructor --- */
Segmentino::Segmentino(float inputSampleRate) :
    Vamp::Plugin(inputSampleRate),
    m_d(0),
    m_chromadata(0),
    m_bpb(4),
    m_pluginFrameCount(0)
{
}


/* --- Desctructor --- */
Segmentino::~Segmentino()
{
    delete m_d;
    delete m_chromadata;
}


/* --- Methods --- */
string Segmentino::getIdentifier() const
{
    return "segmentino";
}

string Segmentino::getName() const
{
    return "Segmentino";
}

string Segmentino::getDescription() const
{
    return "Estimate contiguous segments pertaining to song parts such as verse and chorus.";
}

string Segmentino::getMaker() const
{
    return "Queen Mary, University of London";
}

int Segmentino::getPluginVersion() const
{
    return 2;
}

string Segmentino::getCopyright() const
{
    return "Plugin by Matthew Davies, Christian Landone, Chris Cannam, Matthias Mauch and Massimiliano Zanoni  Copyright (c) 2006-2013 QMUL - Affero GPL";
}

Segmentino::ParameterList Segmentino::getParameterDescriptors() const
{
    ParameterList list;

    ParameterDescriptor desc;

    // desc.identifier = "bpb";
    // desc.name = "Beats per Bar";
    // desc.description = "The number of beats in each bar";
    // desc.minValue = 2;
    // desc.maxValue = 16;
    // desc.defaultValue = 4;
    // desc.isQuantized = true;
    // desc.quantizeStep = 1;
    // list.push_back(desc);

    return list;
}

float Segmentino::getParameter(std::string name) const
{
    if (name == "bpb") return m_bpb;
    return 0.0;
}

void Segmentino::setParameter(std::string name, float value)
{
    if (name == "bpb") m_bpb = lrintf(value);
}


// Return the StepSize for Chroma Extractor 
size_t Segmentino::getPreferredStepSize() const
{
    size_t step = size_t(m_inputSampleRate * m_stepSecs + 0.0001);
    if (step < 1) step = 1;

    return step;
}

// Return the BlockSize for Chroma Extractor 
size_t Segmentino::getPreferredBlockSize() const
{
    int theoretical = getPreferredStepSize() * 2;
    theoretical *= m_chromaFramesizeFactor; 
    return MathUtilities::nextPowerOfTwo(theoretical);
}


// Initialize the plugin and define Beat Tracker and Chroma Extractor Objects
bool Segmentino::initialise(size_t channels, size_t stepSize, size_t blockSize)
{
    if (m_d) {
        delete m_d;
        m_d = 0;
    }
    if (m_chromadata) {
        delete m_chromadata;
        m_chromadata = 0;
    }

    if (channels < getMinChannelCount() ||
        channels > getMaxChannelCount()) {
        std::cerr << "Segmentino::initialise: Unsupported channel count: "
                  << channels << std::endl;
        return false;
    }

    if (stepSize != getPreferredStepSize()) {
        std::cerr << "ERROR: Segmentino::initialise: Unsupported step size for this sample rate: "
                  << stepSize << " (wanted " << (getPreferredStepSize()) << ")" << std::endl;
        return false;
    }

    if (blockSize != getPreferredBlockSize()) {
        std::cerr << "WARNING: Segmentino::initialise: Sub-optimal block size for this sample rate: "
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
    
    // definition of outputs numbers used internally
    int outputCounter = 1;
    m_beatOutputNumber = outputCounter++;
    m_barsOutputNumber = outputCounter++;
    m_beatcountsOutputNumber = outputCounter++;
    m_beatsdOutputNumber = outputCounter++;
    m_logscalespecOutputNumber = outputCounter++;
    m_bothchromaOutputNumber = outputCounter++;
    m_qchromafwOutputNumber = outputCounter++;    
    m_qchromaOutputNumber = outputCounter++;

    return true;
}

void Segmentino::reset()
{
    if (m_d) m_d->reset();
    if (m_chromadata) m_chromadata->reset();
    m_pluginFrameCount = 0;
}

Segmentino::OutputList Segmentino::getOutputDescriptors() const
{

    OutputList list;

    OutputDescriptor segm;
    segm.identifier = "segmentation";
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
    segm.sampleRate = 1.0 / m_stepSecs;
    segm.hasDuration = true;
    m_segmOutputNumber = 0;

    list.push_back(segm);

    return list;
}

// Executed for each frame - called from the host  

// We use time domain input, because DownBeat requires it -- so we
// use the time-domain version of DetectionFunction::process which
// does its own FFT.  It requires doubles as input, so we need to
// make a temporary copy

// We only support a single input channel
Segmentino::FeatureSet Segmentino::process(const float *const *inputBuffers,
                                           Vamp::RealTime timestamp)
{
    if (!m_d) {
        cerr << "ERROR: Segmentino::process: "
             << "Segmentino has not been initialised"
             << endl;
        return FeatureSet();
    }

    const int fl = m_d->dfConfig.frameLength;

    int sampleOffset = ((m_chromaFramesizeFactor-1) * fl) / 2;
    
    double *dfinput = new double[fl];

    // Since chroma needs a much longer frame size, we only ever use the very
    // beginning of the frame for beat tracking.
    for (int i = 0; i < fl; ++i) dfinput[i] = inputBuffers[0][i];
    double output = m_d->df->processTimeDomain(dfinput);

    delete[] dfinput;

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
    
        float *windowedBuffers = new float[m_chromadata->blockSize];

        m_chromadata->window.cut(&inputBuffers[0][0], &windowedBuffers[0]);
    
        // adjust timestamp (we want the middle of the frame)
        timestamp = timestamp + 
            Vamp::RealTime::frame2RealTime(sampleOffset, lrintf(m_inputSampleRate));

        m_chromadata->baseProcess(&windowedBuffers[0], timestamp);

        delete[] windowedBuffers;
    }

    m_pluginFrameCount++;
    
    FeatureSet fs;
    return fs;
}

Segmentino::FeatureSet Segmentino::getRemainingFeatures()
{
    if (!m_d) {
        cerr << "ERROR: Segmentino::getRemainingFeatures: "
             << "Segmentino has not been initialised"
             << endl;
        return FeatureSet();
    }

    FeatureSet masterFeatureset;
    FeatureSet internalFeatureset = beatTrack();

    int beatcount = internalFeatureset[m_beatOutputNumber].size();
    if (beatcount == 0) return Segmentino::FeatureSet();
    Vamp::RealTime last_beattime = internalFeatureset[m_beatOutputNumber][beatcount-1].timestamp;

    // // THIS FOLLOWING BIT IS WEIRD! REPLACES BEAT-TRACKED BEATS WITH 
    // // UNIFORM 0.5 SEC BEATS
    // internalFeatureset[m_beatOutputNumber].clear();
    // Vamp::RealTime beattime = Vamp::RealTime::fromSeconds(1.0);
    // while (beattime < last_beattime)
    // {
    //     Feature beatfeature;
    //     beatfeature.hasTimestamp = true;
    //     beatfeature.timestamp = beattime;
    //     masterFeatureset[m_beatOutputNumber].push_back(beatfeature);
    //     beattime = beattime + Vamp::RealTime::fromSeconds(0.5);
    // }
    
    FeatureList chromaList = chromaFeatures();
    
    for (int i = 0; i < (int)chromaList.size(); ++i)
    {
        internalFeatureset[m_bothchromaOutputNumber].push_back(chromaList[i]);
    }
    
    // quantised and pseudo-quantised (beat-wise) chroma
    std::vector<FeatureList> quantisedChroma = beatQuantiser(chromaList, internalFeatureset[m_beatOutputNumber]);

    if (quantisedChroma.empty()) return masterFeatureset;
    
    internalFeatureset[m_qchromafwOutputNumber] = quantisedChroma[0];
    internalFeatureset[m_qchromaOutputNumber] = quantisedChroma[1];
    
    // Segmentation
    try {
        masterFeatureset[m_segmOutputNumber] = runSegmenter(quantisedChroma[1]);
    } catch (std::bad_alloc &a) {
        cerr << "ERROR: Segmentino::getRemainingFeatures: Failed to run segmenter, not enough memory (song too long?)" << endl;
    }
    
    return(masterFeatureset);
}

/* ------ Beat Tracker ------ */

Segmentino::FeatureSet Segmentino::beatTrack()
{
    vector<double> df;
    vector<double> beatPeriod;
    vector<double> tempi;
    
    for (int i = 2; i < (int)m_d->dfOutput.size(); ++i) { // discard first two elts
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

    for (int i = 0; i < (int)beats.size(); ++i) {
        
        int frame = beats[i] * m_d->dfConfig.stepSize;
        
        if (dbi < (int)downbeats.size() && i == downbeats[dbi]) {
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
        
        if (i > 0 && i <= (int)beatsd.size()) {
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

Segmentino::FeatureList Segmentino::chromaFeatures()
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
//    cerr << endl << "[NNLS Chroma Plugin] Tuning Log-Frequency Spectrogram ... ";
             
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
//    cerr << "done." << endl;    
    /** Semitone spectrum and chromagrams
        Semitone-spaced log-frequency spectrum derived 
        from the tuned log-freq spectrum above. the spectrum
        is inferred using a non-negative least squares algorithm.
        Three different kinds of chromagram are calculated, "treble", "bass", and "both" (which means 
        bass and treble stacked onto each other).
    **/
/*
    if (m_chromadata->useNNLS == 0) {
        cerr << "[NNLS Chroma Plugin] Mapping to semitone spectrum and chroma ... ";
    } else {
        cerr << "[NNLS Chroma Plugin] Performing NNLS and mapping to chroma ... ";
    }
*/    
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
//    cerr << "done." << endl;

    return returnFeatureList;     
}

/* ------ Beat Quantizer ------ */

std::vector<Vamp::Plugin::FeatureList>
Segmentino::beatQuantiser(Vamp::Plugin::FeatureList chromagram, Vamp::Plugin::FeatureList beats)
{
    std::vector<FeatureList> returnVector;
    
    FeatureList fwQchromagram; // frame-wise beat-quantised chroma
    FeatureList bwQchromagram; // beat-wise beat-quantised chroma


    size_t nChromaFrame = chromagram.size();
    size_t nBeat = beats.size();
    
    if (nBeat == 0 && nChromaFrame == 0) return returnVector;
    
    int nBin = chromagram[0].values.size();
    
    vector<float> tempChroma = vector<float>(nBin);
    
    Vamp::RealTime beatTimestamp = Vamp::RealTime::zeroTime;
    int currBeatCount = -1; // start before first beat
    int framesInBeat = 0;
    
    for (size_t iChroma = 0; iChroma < nChromaFrame; ++iChroma)
    {
        Vamp::RealTime frameTimestamp = chromagram[iChroma].timestamp;
        Vamp::RealTime newBeatTimestamp;
                
        if (currBeatCount != (int)beats.size() - 1) {
            newBeatTimestamp = beats[currBeatCount+1].timestamp;
        } else {
            newBeatTimestamp = chromagram[nChromaFrame-1].timestamp;
        }
                
        if (frameTimestamp > newBeatTimestamp ||
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
            bwQchromaFrame.duration = newBeatTimestamp - beatTimestamp;
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
            beatTimestamp = newBeatTimestamp;
            for (int i = 0; i < nBin; ++i) tempChroma[i] = 0; // average
            framesInBeat = 0;
        }
        framesInBeat++;
        for (int i = 0; i < nBin; ++i) tempChroma[i] += chromagram[iChroma].values[i];
    }
    returnVector.push_back(fwQchromagram);
    returnVector.push_back(bwQchromagram);
    return returnVector;
}



/* -------------------------------- */
/* ------ Support Functions  ------ */
/* -------------------------------- */

// one-dimesion median filter
vec medfilt1(vec v, int medfilt_length)
{    
    // TODO: check if this works with odd and even medfilt_length !!!
    int halfWin = medfilt_length/2;
    
    // result vector
    vec res = zeros<vec>(v.size());
    
    // padding 
    vec padV = zeros<vec>(v.size()+medfilt_length-1);
    
    for (int i=medfilt_length/2; i < medfilt_length/2+(int)v.size(); ++ i)
    {
        padV(i) = v(i-medfilt_length/2);
    }
    
    // the above loop leaves the boundaries at 0, 
    // the two loops below fill them with the start or end values of v at start and end
    for (int i = 0; i < halfWin; ++i) padV(i) = v(0);
    for (int i = halfWin+(int)v.size(); i < (int)v.size()+2*halfWin; ++i) padV(i) = v(v.size()-1);
    
    
    
    // Median filter
    vec win = zeros<vec>(medfilt_length);
    
    for (int i=0; i < (int)v.size(); ++i)
    {
        win = padV.subvec(i,i+halfWin*2);
        win = sort(win);
        res(i) = win(halfWin);
    }
    
    return res;
}


// Quantile
double quantile(vec v, double p)
{
    vec sortV = sort(v);
    int n = sortV.size();
    vec x = zeros<vec>(n+2);
    vec y = zeros<vec>(n+2);
    
    x(0) = 0;
    x(n+1) = 100; 
    
    for (int i=1; i<n+1; ++i)
        x(i) = 100*(0.5+(i-1))/n;
        
    y(0) = sortV(0);
    y.subvec(1,n) = sortV;
    y(n+1) = sortV(n-1);
    
    uvec x2index = find(x>=p*100);
    
    // Interpolation
    double x1 = x(x2index(0)-1);
    double x2 = x(x2index(0));
    double y1 = y(x2index(0)-1);
    double y2 = y(x2index(0));
    
    double res = (y2-y1)/(x2-x1)*(p*100-x1)+y1;
    
    return res;
}

// Max Filtering
mat maxfilt1(mat inmat, int len)
{
    mat outmat = inmat;
    
    for (int i=0; i < (int)inmat.n_rows; ++i)
    {
        if (sum(inmat.row(i)) > 0)
        {
            // Take a window of rows
            int startWin;
            int endWin;
            
            if (0 > i-len)
                startWin = 0;
            else
                startWin = i-len;
            
            if ((int)inmat.n_rows-1 < i+len-1)
                endWin = inmat.n_rows-1;
            else
                endWin = i+len-1;
    
            outmat(i,span::all) = 
                max(inmat(span(startWin,endWin),span::all));
        }
    }
    
    return outmat;
    
}

// Null Parts
Part nullpart(vector<Part> parts, vec barline)
{
    uvec nullindices = ones<uvec>(barline.size());
    for (int iPart=0; iPart<(int)parts.size(); ++iPart)
    {
        //for (int iIndex=0; iIndex < parts[0].indices.size(); ++iIndex) 
        for (int iIndex=0; iIndex < (int)parts[iPart].indices.size(); ++iIndex) 
            for (int i=0; i<parts[iPart].n; ++i) 
            {
                int ind = parts[iPart].indices[iIndex]+i;
                nullindices(ind) = 0;
            }
    }

    Part newPart;
    newPart.n = 1;
    uvec q = find(nullindices > 0);
    
    for (int i=0; i<(int)q.size();++i) 
        newPart.indices.push_back(q(i));

    newPart.letter = '-';
    newPart.value = 0;
    newPart.level = 0;
    
    return newPart;    
}


// Merge Nulls
void mergenulls(vector<Part> &parts)
{
    for (int iPart=0; iPart<(int)parts.size(); ++iPart)
    {
        
        vector<Part> newVectorPart;
        
        if (parts[iPart].letter.compare("-")==0)
        {
            sort (parts[iPart].indices.begin(), parts[iPart].indices.end());
            int newpartind = -1;
            
            vector<int> indices;
            indices.push_back(-2);
            
            for (int iIndex=0; iIndex<(int)parts[iPart].indices.size(); ++iIndex) 
                indices.push_back(parts[iPart].indices[iIndex]);
            
            for (int iInd=1; iInd < (int)indices.size(); ++iInd)
            { 
                if (indices[iInd] - indices[iInd-1] > 1)
                {
                    newpartind++;

                    Part newPart;
                    newPart.letter = 'N';
                    std::stringstream out;
                    out << newpartind+1;
                    newPart.letter.append(out.str());
                    // newPart.value = 20+newpartind+1;
                    newPart.value = 0;
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
            
            for (int i=0; i<(int)newVectorPart.size(); ++i)
                parts.push_back(newVectorPart[i]);
        }
    }
}

/* ------ Segmentation ------ */

vector<Part> songSegment(Vamp::Plugin::FeatureList quantisedChromagram)
{
    
    
    /* ------ Parameters ------ */
    double thresh_beat = 0.85;
    double thresh_seg = 0.80;
    int medfilt_length = 5; 
    int minlength = 28;
    int maxlength = 2*128;
    double quantilePerc = 0.1;
    /* ------------------------ */
    
    
    // Collect Info
    int nBeat = quantisedChromagram.size();                      // Number of feature vector
    int nFeatValues = quantisedChromagram[0].values.size();      // Number of values for each feature vector
    
    if (nBeat < minlength) {
        // return a single part
        vector<Part> parts;
        Part newPart;
        newPart.n = 1;
        newPart.indices.push_back(0);
        newPart.letter = "n1";
        newPart.value = 20;
        newPart.level = 0;
        parts.push_back(newPart);
        return parts;
    }

    irowvec timeStamp = zeros<imat>(1,nBeat);       // Vector of Time Stamps
    
    // Save time stamp as a Vector
    if (quantisedChromagram[0].hasTimestamp)
    {
        for (int i = 0; i < nBeat; ++ i)
            timeStamp[i] = quantisedChromagram[i].timestamp.nsec;
    }
    
    
    // Build a ObservationTOFeatures Matrix
    mat featVal = zeros<mat>(nBeat,nFeatValues/2);
    
    for (int i = 0; i < nBeat; ++ i)
        for (int j = 0; j < nFeatValues/2; ++ j)
        {
            featVal(i,j) = 0.8 * quantisedChromagram[i].values[j] + quantisedChromagram[i].values[j+12]; // bass attenuated
        }
    
    // Set to arbitrary value to feature vectors with low std
    mat a = stddev(featVal,1,1);
    
    // Feature Correlation Matrix
    mat simmat0 = 1-cor(trans(featVal));
    

    for (int i = 0; i < nBeat; ++ i)
    {
        if (a(i)<0.000001)
        {
            featVal(i,1) = 1000;  // arbitrary  
            
            for (int j = 0; j < nFeatValues/2; ++j)
            {
                simmat0(i,j) = 1;
                simmat0(j,i) = 1;
            }
        }
    }
    
    mat simmat = 1-simmat0/2;
    
    // -------- To delate when the proble with the add of beat will be solved -------
    for (int i = 0; i < nBeat; ++ i)
     for (int j = 0; j < nBeat; ++ j)
         if (!std::isfinite(simmat(i,j)))
             simmat(i,j)=0;
    // ------------------------------------------------------------------------------
    
    // Median Filtering applied to the Correlation Matrix
    // The median filter is for each diagonal of the Matrix
    mat median_simmat = zeros<mat>(nBeat,nBeat);
    
    for (int i = 0; i < nBeat; ++ i)
    {
        vec temp = medfilt1(simmat.diag(i),medfilt_length);
        median_simmat.diag(i) = temp;
        median_simmat.diag(-i) = temp;
    }

    for (int i = 0; i < nBeat; ++ i)
        for (int j = 0; j < nBeat; ++ j)
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
    uvec dup = find(median_simmat > thresh_beat);
    mat potential_duplicates = zeros<mat>(nBeat,nBeat);
    potential_duplicates.elem(dup) = ones<vec>(dup.size());
    potential_duplicates = trimatu(potential_duplicates);
    
    int nPartlengths = round((maxlength-minlength)/4)+1;
    vec partlengths = zeros<vec>(nPartlengths);
    
    for (int i = 0; i < nPartlengths; ++ i)
        partlengths(i) = (i*4) + minlength;
    
    // initialise arrays
    cube simArray = zeros<cube>(nBeat,nBeat,nPartlengths);
    cube decisionArray2 = zeros<cube>(nBeat,nBeat,nPartlengths);

    for (int iLength = 0; iLength < nPartlengths; ++ iLength)
    // for (int iLength = 0; iLength < 20; ++ iLength)
    {
        int len = partlengths(iLength);
        int nUsedBeat = nBeat - len + 1;                   // number of potential rep beginnings: they can't overlap at the end of the song

        if (nUsedBeat < 1) continue;
        
        for (int iBeat = 0; iBeat < nUsedBeat; ++ iBeat)   // looping over all columns (arbitrarily chosen columns)
        {
            uvec help2 = find(potential_duplicates(span(0,nUsedBeat-1),iBeat)==1);
            
            for (int i=0; i < (int)help2.size(); ++i)
            {

                // measure how well two length len segments go together
                int kBeat = help2(i);
                vec distrib = median_simmat(span(iBeat,iBeat+len-1), span(kBeat,kBeat+len-1)).diag(0);
                simArray(iBeat,kBeat,iLength) = quantile(distrib,quantilePerc);
            }
        }
        
        mat tempM = simArray(span(0,nUsedBeat-1), span(0,nUsedBeat-1), span(iLength,iLength));
        simArray.slice(iLength)(span(0,nUsedBeat-1), span(0,nUsedBeat-1)) = tempM + trans(tempM) - (eye<mat>(nUsedBeat,nUsedBeat)%tempM); 
        
        // convolution
        vec K = zeros<vec>(3);
        K << 0.01 << 0.98 << 0.01;
        
        
        for (int i=0; i < (int)simArray.n_rows; ++i)
        {
            rowvec t = conv((rowvec)simArray.slice(iLength).row(i),K);
            simArray.slice(iLength)(i, span::all) = t.subvec(1,t.size()-2);
        }
 
        // take only over-average bars that do not overlap
        
        mat temp = zeros<mat>(simArray.n_rows, simArray.n_cols);
        temp(span::all, span(0,nUsedBeat-1)) = simArray.slice(iLength)(span::all, span(0,nUsedBeat-1));
        
        for (int i=0; i < (int)temp.n_rows; ++i)
            for (int j=0; j < nUsedBeat; ++j)
                if (temp(i,j) < thresh_seg)
                    temp(i,j) = 0;
        
        decisionArray2.slice(iLength) = temp;

        mat maxMat = maxfilt1(decisionArray2.slice(iLength),len-1);
        
        for (int i=0; i < (int)decisionArray2.n_rows; ++i)
            for (int j=0; j < (int)decisionArray2.n_cols; ++j)
                if (decisionArray2.slice(iLength)(i,j) < maxMat(i,j))
                    decisionArray2.slice(iLength)(i,j) = 0;
        
        decisionArray2.slice(iLength) = decisionArray2.slice(iLength) % trans(decisionArray2.slice(iLength));
        
        for (int i=0; i < (int)simArray.n_rows; ++i)
            for (int j=0; j < (int)simArray.n_cols; ++j)
                if (simArray.slice(iLength)(i,j) < thresh_seg)
                    potential_duplicates(i,j) = 0; 
    }
    
    // Milk the data
    
    mat bestval;
    
    for (int iLength=0; iLength<nPartlengths; ++iLength)
    {
        mat temp = zeros<mat>(decisionArray2.n_rows,decisionArray2.n_cols);

       for (int rows=0; rows < (int)decisionArray2.n_rows; ++rows)
            for (int cols=0; cols < (int)decisionArray2.n_cols; ++cols)
                if (decisionArray2.slice(iLength)(rows,cols) > 0)
                    temp(rows,cols) = 1;
        
        vec currLogicSum = sum(temp,1);
        
        for (int iBeat=0; iBeat < nBeat; ++iBeat)
            if (currLogicSum(iBeat) > 1)
            {
                vec t = decisionArray2.slice(iLength)(span::all,iBeat);
                double currSum = sum(t);
                
                int count = 0;
                for (int i=0; i < (int)t.size(); ++i)
                    if (t(i)>0)
                        count++;
                
                currSum = (currSum/count)/2;
                
                rowvec t1;
                t1 << (currLogicSum(iBeat)-1) * partlengths(iLength) << currSum << iLength << iBeat << currLogicSum(iBeat);
                
                bestval = join_cols(bestval,t1);
            }
    }
    
    // Definition of the resulting vector
    vector<Part> parts;
    
    // make a table of all valid sets of parts
    
    char partletters[] = {'A','B','C','D','E','F','G', 'H','I','J','K','L','M','N','O','P','Q','R','S'};
    int partvalues[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
    vec valid_sets = ones<vec>(bestval.n_rows);
    
    if (!bestval.is_empty())
    {
        
        // In questo punto viene introdotto un errore alla 3 cifra decimale
        
        colvec t = zeros<colvec>(bestval.n_rows);
        for (int i=0; i < (int)bestval.n_rows; ++i)
        {
            t(i) = bestval(i,1)*2;
        }
        
        double m = t.max();
        
        bestval(span::all,1) = bestval(span::all,1) / m; 
        bestval(span::all,0) = bestval(span::all,0) + bestval(span::all,1);
        
        mat bestval2;
        for (int i=0; i < (int)bestval.n_cols; ++i)
            if (i!=1)
                bestval2 = join_rows(bestval2,bestval.col(i));
        
        for (int kSeg=0; kSeg<6; ++kSeg)
        {
            mat currbestvals = zeros<mat>(bestval2.n_rows, bestval2.n_cols);
            for (int i=0; i < (int)bestval2.n_rows; ++i)
                for (int j=0; j < (int)bestval2.n_cols; ++j)
                    if (valid_sets(i))
                        currbestvals(i,j) = bestval2(i,j);
            
            vec t1 = currbestvals.col(0);
            double ma;
            uword maIdx;
            ma = t1.max(maIdx);
            
            if ((maIdx == 0)&&(ma == 0))
                break;

            int bestLength = lrint(partlengths(currbestvals(maIdx,1)));
            rowvec bestIndices = decisionArray2.slice(currbestvals(maIdx,1))(currbestvals(maIdx,2), span::all);
                
            rowvec bestIndicesMap = zeros<rowvec>(bestIndices.size());
            for (int i=0; i < (int)bestIndices.size(); ++i)
                if (bestIndices(i)>0)
                    bestIndicesMap(i) = 1;
                   
            rowvec mask = zeros<rowvec>(bestLength*2-1);
            for (int i=0; i<bestLength; ++i)
                mask(i+bestLength-1) = 1;
            
            rowvec t2 = conv(bestIndicesMap,mask); 
            rowvec island = t2.subvec(mask.size()/2,t2.size()-1-mask.size()/2);
            
            // Save results in the structure
            Part newPart;
            newPart.n = bestLength;
            uvec q1 = find(bestIndices > 0);
            
            for (int i=0; i < (int)q1.size();++i)
                newPart.indices.push_back(q1(i));
            
            newPart.letter = partletters[kSeg];
            newPart.value = partvalues[kSeg];
            newPart.level = kSeg+1;
            parts.push_back(newPart);
            
            uvec q2 = find(valid_sets==1);
            
            for (int i=0; i < (int)q2.size(); ++i)
            {
                int iSet = q2(i);
                int s = partlengths(bestval2(iSet,1));
                
                rowvec mask1 = zeros<rowvec>(s*2-1);
                for (int i=0; i<s; ++i)
                    mask1(i+s-1) = 1;
                
                rowvec Ind = decisionArray2.slice(bestval2(iSet,1))(bestval2(iSet,2), span::all);
                rowvec IndMap = zeros<rowvec>(Ind.size());
                for (int i=0; i < (int)Ind.size(); ++i)
                    if (Ind(i)>0)
                        IndMap(i) = 2;
                
                rowvec t3 = conv(IndMap,mask1); 
                rowvec currislands = t3.subvec(mask1.size()/2,t3.size()-1-mask1.size()/2);       
                rowvec islandsdMult = currislands%island;
                
                uvec islandsIndex = find(islandsdMult > 0);
                
                if (islandsIndex.size() > 0)
                    valid_sets(iSet) = 0;
            }
        }
    }
    else
    {
        Part newPart;
        newPart.n = nBeat;
        newPart.indices.push_back(0);
        newPart.letter = 'A';
        newPart.value = 1;
        newPart.level = 1;
        parts.push_back(newPart);
    }
   
    vec bar = linspace(1,nBeat,nBeat);    
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



void songSegmentChroma(Vamp::Plugin::FeatureList quantisedChromagram, vector<Part> &parts)
{
    // Collect Info
    int nBeat = quantisedChromagram.size();                      // Number of feature vector
    int nFeatValues = quantisedChromagram[0].values.size();      // Number of values for each feature vector

    mat synchTreble = zeros<mat>(nBeat,nFeatValues/2);
    
    for (int i = 0; i < nBeat; ++ i)
        for (int j = 0; j < nFeatValues/2; ++ j)
        {
            synchTreble(i,j) = quantisedChromagram[i].values[j];
        }
    
    mat synchBass = zeros<mat>(nBeat,nFeatValues/2);
    
    for (int i = 0; i < nBeat; ++ i)
        for (int j = 0; j < nFeatValues/2; ++ j)
        {
            synchBass(i,j) = quantisedChromagram[i].values[j+12];
        }

    // Process
    
    mat segTreble = zeros<mat>(quantisedChromagram.size(),quantisedChromagram[0].values.size()/2);
    mat segBass = zeros<mat>(quantisedChromagram.size(),quantisedChromagram[0].values.size()/2);
    
    for (int iPart=0; iPart < (int)parts.size(); ++iPart)
    {
        parts[iPart].nInd = parts[iPart].indices.size();
        
        for (int kOccur=0; kOccur<parts[iPart].nInd; ++kOccur)
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
    
    for (int iPart=0; iPart < (int)parts.size(); ++iPart)
    {
        parts[iPart].nInd = parts[iPart].indices.size();
        for (int iInstance=0; iInstance<parts[iPart].nInd; ++iInstance)
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
    
    for (int i=0; i < (int)partindices.size(); ++i)
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
    for (int iNewpart=1; iNewpart < (int)newPartVector.size(); ++iNewpart)
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
Vamp::Plugin::FeatureList Segmentino::runSegmenter(Vamp::Plugin::FeatureList quantisedChromagram)
{
    /* --- Display Information --- */
//    int numBeat = quantisedChromagram.size();
//    int numFeats = quantisedChromagram[0].values.size();

    vector<Part> parts;
    vector<Part> finalParts;
    
    parts = songSegment(quantisedChromagram);
    songSegmentChroma(quantisedChromagram,parts);
    
    finalParts = songSegmentIntegration(parts);
    
    
    // TEMP ----
    /*for (int i=0;i<finalParts.size(); ++i)
     {
     std::cout << "Parts n° " << i << std::endl;
     std::cout << "n°: " << finalParts[i].n << std::endl;
     std::cout << "letter: " <<  finalParts[i].letter << std::endl;
     
     std::cout << "indices: ";
     for (int j=0;j<finalParts[i].indices.size(); ++j)
         std::cout << finalParts[i].indices[j] << " ";
       
     std::cout << std::endl;
     std::cout <<  "level: " << finalParts[i].level << std::endl;
     }*/
    
    // ---------
    
    
    // Output

    Vamp::Plugin::FeatureList results;
    
    
    Feature seg;
    
    vec indices;
//    int idx=0;
    vector<int> values;
    vector<string> letters;
    
    for (int iPart=0; iPart < (int)finalParts.size()-1; ++iPart)
    {
        int iInstance=0;
        seg.hasTimestamp = true;
         
        int ind = finalParts[iPart].indices[iInstance];
        int ind1 = finalParts[iPart+1].indices[iInstance];
         
        seg.timestamp = quantisedChromagram[ind].timestamp;
        seg.hasDuration = true;
        seg.duration = quantisedChromagram[ind1].timestamp-quantisedChromagram[ind].timestamp;
        seg.values.clear();
        seg.values.push_back(finalParts[iPart].value);
        seg.label = finalParts[iPart].letter;
         
        results.push_back(seg);
    }
    
    if (finalParts.size() > 0) {
        int ind = finalParts[finalParts.size()-1].indices[0];
        seg.hasTimestamp = true;
        seg.timestamp = quantisedChromagram[ind].timestamp;
        seg.hasDuration = true;
        seg.duration = quantisedChromagram[quantisedChromagram.size()-1].timestamp-quantisedChromagram[ind].timestamp;
        seg.values.clear();
        seg.values.push_back(finalParts[finalParts.size()-1].value);
        seg.label = finalParts[finalParts.size()-1].letter;
    
        results.push_back(seg);
    }

    return results;    
}

















