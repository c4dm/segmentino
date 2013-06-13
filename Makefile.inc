
PLUGIN_EXT	?= .so
PLUGIN	?= segmentino$(PLUGIN_EXT)
CXX	?= g++
CC	?= gcc

CFLAGS		:= $(CFLAGS) -I. -Ivamp-plugin-sdk -Iqm-dsp -Iarmadillo-3.900.4/include
CXXFLAGS	:= $(CXXFLAGS) $(CFLAGS)
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
	   qm-dsp/maths/MathUtilities.cpp

OBJECTS := $(SOURCES:.cpp=.o)
OBJECTS := $(OBJECTS:.c=.o)

$(PLUGIN):	$(OBJECTS)
		$(CXX) -o $@ $^ $(LDFLAGS)

clean:		
		rm $(OBJECTS)

distclean:	clean
		rm $(PLUGIN)
