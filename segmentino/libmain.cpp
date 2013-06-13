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

#include <vamp/vamp.h>
#include <vamp-sdk/PluginAdapter.h>

#include "Segmentino.h"

static Vamp::PluginAdapter<Segmentino> segmentinoAdapter;

const VampPluginDescriptor *vampGetPluginDescriptor(unsigned int vampApiVersion,
                                                    unsigned int index)
{
    if (vampApiVersion < 1) return 0;

    switch (index) {
    case 0: return segmentinoAdapter.getDescriptor();
    default: return 0;
    }
}

