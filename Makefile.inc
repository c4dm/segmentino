
PLUGIN_EXT	?= .so
PLUGIN	?= segmentino$(PLUGIN_EXT)
CXX	?= g++
CC	?= gcc

INCLUDEFLAGS	:= -I. -Ivamp-plugin-sdk -Iqm-dsp -Iqm-dsp/ext/kissfft -Iqm-dsp/ext/kissfft/tools -Dkiss_fft_scalar=double -Iarmadillo-3.900.4/include

CFLAGS		:= $(CFLAGS) $(INCLUDEFLAGS)
CXXFLAGS	:= $(CXXFLAGS) $(INCLUDEFLAGS)
LDFLAGS		:= $(LDFLAGS)

HEADERS := segmentino/Segmentino.h

SOURCES := segmentino/Segmentino.cpp \
           segmentino/libmain.cpp \
           nnls-chroma/chromamethods.cpp \
           nnls-chroma/nnls.c \
	   vamp-plugin-sdk/src/vamp-sdk/PluginAdapter.cpp \
	   vamp-plugin-sdk/src/vamp-sdk/RealTime.cpp \
	   qm-dsp/dsp/onsets/DetectionFunction.cpp \
	   qm-dsp/dsp/onsets/PeakPicking.cpp \
	   qm-dsp/dsp/transforms/FFT.cpp \
	   qm-dsp/dsp/rateconversion/Decimator.cpp \
	   qm-dsp/dsp/tempotracking/TempoTrackV2.cpp \
	   qm-dsp/dsp/tempotracking/DownBeat.cpp \
	   qm-dsp/dsp/phasevocoder/PhaseVocoder.cpp \
           qm-dsp/dsp/signalconditioning/DFProcess.cpp \
           qm-dsp/dsp/signalconditioning/FiltFilt.cpp \
           qm-dsp/dsp/signalconditioning/Filter.cpp \
	   qm-dsp/maths/MathUtilities.cpp \
           qm-dsp/ext/kissfft/kiss_fft.c \
           qm-dsp/ext/kissfft/tools/kiss_fftr.c

OBJECTS := $(SOURCES:.cpp=.o)
OBJECTS := $(OBJECTS:.c=.o)

$(PLUGIN):	$(OBJECTS)
		$(CXX) -o $@ $^ $(LDFLAGS)

clean:		
		rm -f $(OBJECTS)

distclean:	clean
		rm -f $(PLUGIN)
