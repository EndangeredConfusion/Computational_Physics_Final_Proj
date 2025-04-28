# Computational_Physics_Final_Proj

Here can be found the MacOS compatable GPU code as well as the csv output (need most recent Xcode version).

To run pull the .m and .metal files into the same directory then build the API:
`xcrun -sdk macosx metal -c sphere.metal -o sphere.air `
then
`xcrun -sdk macosx metallib sphere.air -o sphere.metallib`

Compile .m file
`clang -fobjc-arc -framework Foundation -framework Metal main.m -o pi_metal`

Run with `./pi_metal` per usual.

To get a new csv file pipe the output into the file!
`./pi_metal > <my name>.csv`
