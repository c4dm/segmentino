### SEGMENTINO VAMP PLUGIN ###

Segmentino is a Vamp plugin for automatic music structural
segmentation, based on an algorithm first used in Mauch's paper on
"Using Musical Structure to Enhance Automatic Chord Transcription"
(http://code.soundsoftware.ac.uk/publications/55).

## Installation ##

Vamp plugins (http://www.vamp-plugins.org/) can be used with any Vamp
host. Popular free hosts are Sonic Visualiser
(http://sonicvisualiser.org), Audacity
(http://audacity.sourceforge.net/) and Sonic Annotator (command-line
tool, http://www.omras2.org/sonicannotator().

For the newest Segmentino plugin downloads, ready to be installed, go
to the Downloads page:
http://code.soundsoftware.ac.uk/projects/segmenter-vamp-plugin/files.

Installation instructions for Mac, Windows, Linux can be found at:
http://www.vamp-plugins.org/download.html.

## Method Overview ##

The idea of the method is quite simple. A beat-quantised chroma
representation is used to calculate pair-wise similarities between
beats (really: beat "shingles", i.e. multi-beat vectors). Based on
this first similarity calculation, an exhaustive comparison of all
possible segments of reasonable length in beats is executed, and
segments are added to form segment families if they are sufficiently
similar to another "family member".

Now comes the crucial greedy step: having accumulated a lot of
families, the families are rated, and the one with the highest score
is used as the first segmentation group that gets annotated. This last
step is repeated until no more families fit the remaining "holes" in
the song that haven't already been assigned to a segment. This is
described in Mauch's thesis:
http://code.soundsoftware.ac.uk/publications/56 as "heuristic of a
music editor who tries to make a concise transcription to make
orientation in the score easier, and to save space (or paper)".

More details in Mauch's thesis: http://code.soundsoftware.ac.uk/publications/56.

## Limitations ##

This method was developed for "classic rock" music, and therefore
assumes a few characteristics that are not necessarily found in other
music: repetition of harmonic sequences in the music that coincide
with structural segments in a song; a steady beat; segments of a
certain length; corresponding segments have the same length in
beats. Furthermore, as with any audio algorithm, the method doesn't
work perfectly even if these conditions are actually fulfilled.

## Implementation ##

This Vamp implementation was coded by Massimiliano Zanoni, Matthias
Mauch and Chris Cannam. It is similar to Mauch's original Matlab
implementation used in the publications, but not exactly the same. The
beat times are extracted using Davies's beat tracker as implemented in
the Queen Mary Vamp plugins
(http://code.soundsoftware.ac.uk/projects/qm-vamp-plugins), which in
turn is based on the QM DSP library
(http://code.soundsoftware.ac.uk/projects/qm-dsp). NNLS Chroma
(http://isophonics.net/nnls-chroma) is used as a chroma extractor, by
borrowing code from the Vamp plugin of the same name.

## Examples ##

Some example YouTube videos showing off the range of quality of
segmentation one can get using Segmentino.

1. excellent: "Bring On The Night" http://www.youtube.com/v/c0W_n_i3Ksg&autoplay=1
2. very good: "Dirty Work" http://www.youtube.com/v/9lWrvvP4PCw&autoplay=1
3. less than good: "Every Kinda People" http://www.youtube.com/v/ICz-PnhfD0M&autoplay=1
4 complete failure: "New York New York" http://www.youtube.com/v/LcmTCeAKM48&autoplay=1

## Citation, License and Use ##

If you make use of this software for any public or commercial purpose,
we ask you to kindly mention the authors and Queen Mary, University of
London in your user-visible documentation. We're very happy to see
this sort of use but would much appreciate being credited, separately
from the requirements of the software license itself (see below).

If you make use of this software for academic purposes, please cite
one of the publications indicated on the Publications page
https://code.soundsoftware.ac.uk/publications?project_id=segmenter-vamp-plugin.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

## Pronunciation ##

Segmentino is pronounced "segmen'TEEno" (not "seg'MENtino", the third
person plural conjunctive of the Italian verb "segmentare").

