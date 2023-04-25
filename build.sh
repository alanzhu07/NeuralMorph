source config
echo "Torch path: ${TORCH_CMAKE_PATH}"
echo "Juce path: ${JUCE_CMAKE_PATH}"
echo "Generator: ${CMAKE_GENERATOR}"

echo "making build tree"
if [ -d build ]; then rm -r build; fi
cmake -Bbuild -G "${CMAKE_GENERATOR}" -DCMAKE_PREFIX_PATH="${TORCH_CMAKE_PATH};${JUCE_CMAKE_PATH}"

echo "building for release"
cmake --build build --target NeuralMorph_All --config Release