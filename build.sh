TORCH_CMAKE_PATH="/opt/homebrew/Cellar/pytorch/1.13.1/share/cmake"
JUCE_CMAKE_PATH="~/opt/JUCE"

cmake -Bbuild -GXcode -DCMAKE_PREFIX_PATH="${TORCH_CMAKE_PATH};${JUCE_CMAKE_PATH}"