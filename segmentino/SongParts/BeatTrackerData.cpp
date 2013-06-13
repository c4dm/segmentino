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

//#include "SongParts.h"

#include "BeatTrackerData.h"

//#include <base/Window.h>
#include <dsp/onsets/DetectionFunction.h>
//#include <dsp/onsets/PeakPicking.h>
//#include <dsp/transforms/FFT.h>
//#include <dsp/tempotracking/TempoTrackV2.h>
//#include <dsp/tempotracking/DownBeat.h>
//#include <chromamethods.h>
//#include <maths/MathUtilities.h>

// #include <vamp-sdk/Plugin.h>

using std::string;
using std::vector;
using std::cerr;
using std::endl;


//#ifndef __GNUC__
//#include <alloca.h>
//#endif


/* ------------------------------------ */
/* ----- BEAT DETECTOR CLASS ---------- */
/* ------------------------------------ */


/* --- ATTRIBUTES --- */
private:
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

/* --- Getter Methods ---*/
DFConfig getdfConfig(){
return dfConfig;
}









