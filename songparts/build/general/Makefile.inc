
PLUGIN_EXT	?= .so
PLUGIN	?= qm-vamp-plugins$(PLUGIN_EXT)
CXX	?= g++
CC	?= gcc

CFLAGS	:= $(CFLAGS) -I. -I../../qm-dsp -I../../nnls-chroma
CXXFLAGS	:= $(CXXFLAGS) -I. -I../../qm-dsp -I../../nnls-chroma
LDFLAGS		:= -L../../qm-dsp  $(LDFLAGS)

HEADERS := SongParts.h

SOURCES := SongParts.cpp \
           libmain.cpp \
           ../../nnls-chroma/chromamethods.cpp \
           ../../nnls-chroma/nnls.c

OBJECTS := $(SOURCES:.cpp=.o)
OBJECTS := $(OBJECTS:.c=.o)

$(PLUGIN):	$(OBJECTS)
		$(CXX) -o $@ $^ $(LDFLAGS)

clean:		
		rm $(OBJECTS)

distclean:	clean
		rm $(PLUGIN)
