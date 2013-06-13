/* -*- c-basic-offset: 4 indent-tabs-mode: nil -*-  vi:set ts=8 sts=4 sw=4: */

/*
    Segmentino

    Code by Massimiliano Zanoni and Matthias Mauch
    Centre for Digital Music, Queen Mary, University of London

    Copyright 2009-2013 Queen Mary, University of London.

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version.  See the file
    COPYING included with this distribution for more information.
*/

#ifndef _SEGMENTINO_PLUGIN_H_
#define _SEGMENTINO_PLUGIN_H_

#include <vamp-sdk/Plugin.h>

class BeatTrackerData;

class ChromaData;

class Segmentino : public Vamp::Plugin
{
public:
    Segmentino(float inputSampleRate);
    virtual ~Segmentino();

    bool initialise(size_t channels, size_t stepSize, size_t blockSize);
    void reset();

    InputDomain getInputDomain() const { return TimeDomain; }

    std::string getIdentifier() const;
    std::string getName() const;
    std::string getDescription() const;
    std::string getMaker() const;
    int getPluginVersion() const;
    std::string getCopyright() const;

    ParameterList getParameterDescriptors() const;
    float getParameter(std::string) const;
    void setParameter(std::string, float);

    size_t getPreferredStepSize() const;
    size_t getPreferredBlockSize() const;

    OutputList getOutputDescriptors() const;

    FeatureSet process(const float *const *inputBuffers, Vamp::RealTime timestamp);
    FeatureSet getRemainingFeatures();

protected:
    BeatTrackerData *m_d;
    ChromaData *m_chromadata;
    static float m_stepSecs;
    static int m_chromaFramesizeFactor;
    static int m_chromaStepsizeFactor;
    int m_bpb;
    int m_pluginFrameCount;
    FeatureSet beatTrack();
    FeatureList chromaFeatures();
    std::vector<FeatureList> beatQuantiser(FeatureList chromagram, FeatureList beats);
    FeatureList runSegmenter(FeatureList quantisedChromagram);
    
    mutable int m_beatOutputNumber;
    mutable int m_barsOutputNumber;
    mutable int m_beatcountsOutputNumber;
    mutable int m_beatsdOutputNumber;
    mutable int m_logscalespecOutputNumber;
    mutable int m_bothchromaOutputNumber;
    mutable int m_qchromafwOutputNumber;
    mutable int m_qchromaOutputNumber;
    mutable int m_segmOutputNumber;
    
};


#endif
