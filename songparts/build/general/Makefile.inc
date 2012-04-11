
PLUGIN_EXT	?= .so
PLUGIN	?= qm-vamp-plugins$(PLUGIN_EXT)
CXX	?= g++
CC	?= gcc

CFLAGS	:= $(CFLAGS) -I. -I../qm-dsp -I../nnls-chroma
CXXFLAGS	:= $(CXXFLAGS) -I. -I../qm-dsp -I../nnls-chroma
LDFLAGS		:= -L../qm-dsp  $(LDFLAGS)

HEADERS := plugins/AdaptiveSpectrogram.h \
           plugins/BarBeatTrack.h \
           plugins/BeatTrack.h \
           plugins/DWT.h \
           plugins/OnsetDetect.h \
           plugins/ChromagramPlugin.h \
           plugins/ConstantQSpectrogram.h \
           plugins/KeyDetect.h \
           plugins/MFCCPlugin.h \
           plugins/SegmenterPlugin.h \
           plugins/SimilarityPlugin.h \
           plugins/SongParts.h \
           plugins/TonalChangeDetect.h \
           plugins/Transcription.h

SOURCES := g2cstubs.c \
           plugins/AdaptiveSpectrogram.cpp \
           plugins/BarBeatTrack.cpp \
           plugins/BeatTrack.cpp \
           plugins/DWT.cpp \
           plugins/OnsetDetect.cpp \
           plugins/ChromagramPlugin.cpp \
           plugins/ConstantQSpectrogram.cpp \
           plugins/KeyDetect.cpp \
           plugins/MFCCPlugin.cpp \
           plugins/SegmenterPlugin.cpp \
           plugins/SongParts.cpp \
           plugins/SimilarityPlugin.cpp \
           plugins/TonalChangeDetect.cpp \
           plugins/Transcription.cpp \
           libmain.cpp \
		   ../nnls-chroma/chromamethods.cpp \
		   ../nnls-chroma/nnls.c

OBJECTS := $(SOURCES:.cpp=.o)
OBJECTS := $(OBJECTS:.c=.o)

$(PLUGIN):	$(OBJECTS)
		$(CXX) -o $@ $^ $(LDFLAGS)

clean:		
		rm $(OBJECTS)

distclean:	clean
		rm $(PLUGIN)
