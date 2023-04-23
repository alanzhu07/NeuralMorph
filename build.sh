# TORCH_CMAKE_PATH="/opt/homebrew/Cellar/pytorch/1.13.1/share/cmake"
# JUCE_CMAKE_PATH="~/opt/JUCE"

# rm -r build
# cmake -Bbuild -GXcode -DCMAKE_PREFIX_PATH="${TORCH_CMAKE_PATH};${JUCE_CMAKE_PATH}"

# echo "building for release"
# cmake --build build --target NeuralMorph --config Release

cmake --build build --target NeuralMorph_All --config Release