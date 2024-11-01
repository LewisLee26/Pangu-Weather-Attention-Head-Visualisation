document.addEventListener('DOMContentLoaded', async () => {
    // Constants for variable indices
    const surfaceVarIdx = { "MSLP": 0, "U10": 1, "V10": 2, "T2M": 3 };
    const upperVarIdx = { "Z": 0, "Q": 1, "T": 2, "U": 3, "V": 4 };

    // Intermediate layer names
    const intermediateLayerNames = [
        '/b1/Add_output_0', '/b1/Add_3_output_0', '/b1/Add_7_output_0', '/b1/Add_10_output_0',
        '/b1/Add_14_output_0', '/b1/Add_17_output_0', '/b1/Add_21_output_0', '/b1/Add_24_output_0',
        '/b1/Add_28_output_0', '/b1/Add_31_output_0', '/b1/Add_35_output_0', '/b1/Add_38_output_0',
        '/b1/Add_42_output_0', '/b1/Add_45_output_0', '/b1/Add_49_output_0', '/b1/Add_52_output_0',
    ];

    // Current state variables
    let state = {
        currentSurfaceVariable: 'T2M',
        currentUpperVariable: 'Z',
        currentAttentionHead: 0,
        currentLatIndex: 0,
        currentLonIndex: 0,
        currentDate: "",
        currentTime: "",
        currentLayer: "",
        currentPressureLevel: 0
    };

    // Fetch available data
    const availableData = await fetch('available_data.json').then(response => response.json());

    // DOM elements
    const dateSelect = document.getElementById('date-select');
    const timeSelect = document.getElementById('time-select');
    const layerSelect = document.getElementById('layer-select');
    const attentionHeadSelect = document.getElementById('attention-head');

    // Populate date select
    populateSelect(dateSelect, Object.keys(availableData));
    state.currentDate = dateSelect.value;

    // Populate time and layer selects
    function populateTimeAndLayerSelects() {
        populateSelect(timeSelect, Object.keys(availableData[state.currentDate]));
        state.currentTime = timeSelect.value;

        populateSelect(layerSelect, availableData[state.currentDate][state.currentTime]);
        state.currentLayer = layerSelect.value;
        updateAttentionHeadOptions();
    }

    // Populate a select element with options
    function populateSelect(selectElement, options) {
        selectElement.innerHTML = '';
        options.forEach(option => selectElement.add(new Option(option, option)));
    }

    // Update attention head options
    function updateAttentionHeadOptions() {
        attentionHeadSelect.innerHTML = '';
        const { numHeads } = getLayerConfig(state.currentLayer);
        for (let i = 0; i < numHeads; i++) {
            attentionHeadSelect.add(new Option(`Head ${i}`, i));
        }
        state.currentAttentionHead = attentionHeadSelect.value;
    }

    // Initial population of selects
    populateTimeAndLayerSelects();

    // Get layer configuration
    function getLayerConfig(layerName) {
        const modifiedLayerName = layerName.replace(/^_/, '/').replace('_', '/');
        const layerIndex = intermediateLayerNames.indexOf(modifiedLayerName);
        return (layerIndex < 2 || layerIndex >= intermediateLayerNames.length - 2) ?
            { numHeads: 6, configName: 'config_24x48', chunkSize: [24, 48] } :
            { numHeads: 12, configName: 'config_48x96', chunkSize: [48, 96] };
    }

    // Load binary data
    async function loadBinaryData(url, shape, dtype = 'float32') {
        const response = await fetch(url);
        const arrayBuffer = await response.arrayBuffer();
        const typedArray = new Float32Array(arrayBuffer);
        if (typedArray.length !== shape.reduce((a, b) => a * b)) {
            throw new Error(`Mismatch in data size: expected ${shape.reduce((a, b) => a * b)} but got ${typedArray.length}`);
        }
        return tf.tensor(typedArray, shape, dtype);
    }

    // Load map chunk
    async function loadMapChunk(latIndex, lonIndex, pressureLevel) {
        const { configName, chunkSize } = getLayerConfig(state.currentLayer);
        const layerIndex = intermediateLayerNames.indexOf(state.currentLayer.replace(/^_/, '/').replace('_', '/'));
        const isOddLayer = layerIndex % 2 !== 0;
        const configSuffix = isOddLayer ? '_shifted' : '';
        let tensors = [];

        const upperIndicesMap = {
            0: isOddLayer ? [0, 1, 2] : [0],
            1: isOddLayer ? [3, 4, 5, 6] : [1, 2, 3, 4],
            2: isOddLayer ? [7, 8, 9, 10] : [5, 6, 7, 8],
            3: isOddLayer ? [11, 12] : [9, 10, 11, 12]
        };

        const upperIndices = upperIndicesMap[pressureLevel] || [];
        for (const index of upperIndices) {
            const upperUrl = `bin/${state.currentDate}/${state.currentTime}/${configName}_upper_${index}${configSuffix}/map/input_upper_${latIndex * chunkSize[0]}_${lonIndex * chunkSize[1]}.bin`;
            const upperTensor = await loadBinaryData(upperUrl, [5, chunkSize[0], chunkSize[1]]);
            tensors.push(upperTensor.gather([upperVarIdx[state.currentUpperVariable]], 0));
        }

        if (pressureLevel === 0 && !isOddLayer) {
            const surfaceUrl = `bin/${state.currentDate}/${state.currentTime}/${configName}${configSuffix}/map/input_surface_${latIndex * chunkSize[0]}_${lonIndex * chunkSize[1]}.bin`;
            const surfaceTensor = await loadBinaryData(surfaceUrl, [4, chunkSize[0], chunkSize[1]]);
            tensors.push(surfaceTensor.gather([surfaceVarIdx[state.currentSurfaceVariable]], 0));
        }

        return tensors;
    }

    // Load attention chunk
    async function loadAttentionChunk(lon, latPl, head) {
        const { numHeads } = getLayerConfig(state.currentLayer);
        if (head >= numHeads) {
            throw new Error(`Invalid head index: ${head} for layer ${state.currentLayer}`);
        }
        const url = `bin/${state.currentDate}/${state.currentTime}/${state.currentLayer}/attention/attention_${lon}_${latPl}_${head}.bin`;
        return loadBinaryData(url, [144, 144]);
    }

    // Initialize visualizations
    async function initializeVisualizations() {
        try {
            const mapDataTensors = await loadMapChunk(state.currentLatIndex, state.currentLonIndex, state.currentPressureLevel);
            const latPlIndex = state.currentLatIndex + state.currentPressureLevel * (getLayerConfig(state.currentLayer).chunkSize[0] === 24 ? 31 : 16);
            const attentionData = await loadAttentionChunk(state.currentLonIndex, latPlIndex, state.currentAttentionHead);

            const mapDataArray = mapDataTensors.map(tensor => tensor.arraySync());
            const attentionDataArray = attentionData.arraySync();

            clearContainer("#map-container");
            clearContainer("#attention-container");

            mapDataArray.forEach((data, index) => {
                initMap(data.flat(), mapDataTensors[index].shape, index);
            });
            initAttentionPattern(attentionDataArray, mapDataArray);
        } catch (error) {
            console.error('Failed to initialize visualizations:', error);
        }
    }

    // Clear container
    function clearContainer(selector) {
        d3.select(selector).selectAll("*").remove();
    }

    // Initialize map
    function initMap(data, mapShape, index) {
        const svg = d3.select("#map-container").append("svg")
            .attr("viewBox", `0 0 ${mapShape[2] * 15} ${mapShape[1] * 15}`)
            .attr("preserveAspectRatio", "xMidYMid meet")
            .classed("svg-content-responsive", true);

        const dataExtent = d3.extent(data.flat());
        const colorScale = d3.scaleSequential(d3.interpolateTurbo).domain(dataExtent);

        svg.selectAll("rect")
            .data(data.flat())
            .enter().append("rect")
            .attr("x", (d, i) => (i % mapShape[2]) * 15)
            .attr("y", (d, i) => Math.floor(i / mapShape[2]) * 15)
            .attr("width", 15)
            .attr("height", 15)
            .attr("fill", d => d == null || isNaN(d) ? "gray" : colorScale(d))
            .attr("class", "map-cell")
            .attr("data-original-color", d => d == null || isNaN(d) ? "gray" : colorScale(d));
    }

    // Initialize attention pattern
    function initAttentionPattern(attentionData, mapData) {
        const svg = d3.select("#attention-container").append("svg")
            .attr("viewBox", "0 0 864 864")
            .attr("preserveAspectRatio", "xMidYMid meet")
            .classed("svg-content-responsive", true);

        const colorScale = d3.scaleSequential(d3.interpolateViridis).domain(d3.extent(attentionData.flat()));

        svg.selectAll("rect")
            .data(attentionData.flat())
            .enter().append("rect")
            .attr("x", (d, i) => (i % 144) * 6)
            .attr("y", (d, i) => Math.floor(i / 144) * 6)
            .attr("width", 6)
            .attr("height", 6)
            .attr("fill", d => colorScale(d))
            .on("mouseover", function(event, d, i) {
                const index = svg.selectAll("rect").nodes().indexOf(this);
                highlightMapCells(index, mapData);
                d3.select(this).attr("stroke", "black").attr("stroke-width", 2);
            })
            .on("mouseout", function(event, d) {
                d3.selectAll("rect.map-cell").attr("stroke", null);
                d3.select(this).attr("stroke", null);
            });
    }

    // Highlight map cells
    function highlightMapCells(attentionIndex, mapData) {
        const { chunkSize } = getLayerConfig(state.currentLayer);
        const mapWidth = chunkSize[1];
        
        let patchSize = mapWidth === 48 ? 4 : 8;

        const x = (attentionIndex % 144);
        const y = Math.floor(attentionIndex / 144);

        const tar_pl = Math.floor(x / 72); 
        const src_pl = Math.floor(y / 72);

        const tar_lat = (Math.floor(x / 12) % 6) * patchSize;
        const src_lat = (Math.floor(y / 12) % 6) * patchSize;

        const tar_lon = (x % 12) * patchSize;
        const src_lon = (y % 12) * patchSize;

        function mixColors(c1, c2, ratio) {
            const color1 = d3.color(c1);
            const color2 = d3.color(c2);
            return d3.rgb(
                Math.round(color1.r * (1 - ratio) + color2.r * ratio),
                Math.round(color1.g * (1 - ratio) + color2.g * ratio),
                Math.round(color1.b * (1 - ratio) + color2.b * ratio)
            ).toString();
        }

        d3.selectAll("rect.map-cell")
            .attr("fill", (d, i, nodes) => {
                const originalColor = d3.select(nodes[i]).attr("data-original-color");
                const cellLatIndex = Math.floor(i / mapWidth);
                const cellLonIndex = i % mapWidth;
                const blockHeight = mapWidth / 2;

                let inTargetBlock = false;
                let inSourceBlock = false;

                if (state.currentPressureLevel === 0) {
                    const layerIndex = intermediateLayerNames.indexOf(state.currentLayer.replace(/^_/, '/').replace('_', '/'));
                    const isOddLayer = layerIndex % 2 !== 0;
                    if (isOddLayer) {
                        const upperIndexOffset = (tar_pl === 0) ? 0 : 2;
                        const currentUpperIndex = Math.floor(cellLatIndex / blockHeight);

                        if (tar_pl === 0) {
                            inTargetBlock = cellLonIndex >= tar_lon && cellLonIndex < tar_lon + patchSize &&
                                            cellLatIndex >= tar_lat && cellLatIndex < tar_lat + patchSize;
                        } else if (tar_pl === 1 && currentUpperIndex >= 1) {
                            inTargetBlock = cellLonIndex >= tar_lon && cellLonIndex < tar_lon + patchSize &&
                                            (cellLatIndex % blockHeight) >= tar_lat && (cellLatIndex % blockHeight) < tar_lat + patchSize;
                        }

                        const upperIndexOffsetSource = (src_pl === 0) ? 0 : 2;
                        const currentSourceUpperIndex = Math.floor(cellLatIndex / blockHeight);

                        if (src_pl === 0) {
                            inSourceBlock = cellLonIndex >= src_lon && cellLonIndex < src_lon + patchSize &&
                                            cellLatIndex >= src_lat && cellLatIndex < src_lat + patchSize;
                        } else if (src_pl === 1 && currentSourceUpperIndex >= 1) {
                            inSourceBlock = cellLonIndex >= src_lon && cellLonIndex < src_lon + patchSize &&
                                            (cellLatIndex % blockHeight) >= src_lat && (cellLatIndex % blockHeight) < src_lat + patchSize;
                        }
                    } else {
                        if (tar_pl === 0 && cellLatIndex < blockHeight) {
                            inTargetBlock = cellLonIndex >= tar_lon && cellLonIndex < tar_lon + patchSize &&
                                            cellLatIndex >= tar_lat && cellLatIndex < tar_lat + patchSize;
                        } else if (tar_pl === 1 && cellLatIndex >= blockHeight) {
                            inTargetBlock = cellLonIndex >= tar_lon && cellLonIndex < tar_lon + patchSize &&
                                            (cellLatIndex - blockHeight) >= tar_lat && (cellLatIndex - blockHeight) < tar_lat + patchSize;
                        }

                        if (src_pl === 0 && cellLatIndex < blockHeight) {
                            inSourceBlock = cellLonIndex >= src_lon && cellLonIndex < src_lon + patchSize &&
                                            cellLatIndex >= src_lat && cellLatIndex < src_lat + patchSize;
                        } else if (src_pl === 1 && cellLatIndex >= blockHeight) {
                            inSourceBlock = cellLonIndex >= src_lon && cellLonIndex < src_lon + patchSize &&
                                            (cellLatIndex - blockHeight) >= src_lat && (cellLatIndex - blockHeight) < src_lat + patchSize;
                        }
                    }
                } else {
                    const upperIndexOffset = (tar_pl === 0) ? 0 : 2;
                    const currentUpperIndex = Math.floor(cellLatIndex / blockHeight);
                    
                    if ((tar_pl === 0 && currentUpperIndex < 2) || (tar_pl === 1 && currentUpperIndex >= 2)) {
                        inTargetBlock = cellLonIndex >= tar_lon && cellLonIndex < tar_lon + patchSize &&
                                        (cellLatIndex % blockHeight) >= tar_lat && (cellLatIndex % blockHeight) < tar_lat + patchSize;
                    }

                    const upperIndexOffsetSource = (src_pl === 0) ? 0 : 2;
                    const currentSourceUpperIndex = Math.floor(cellLatIndex / blockHeight);

                    if ((src_pl === 0 && currentSourceUpperIndex < 2) || (src_pl === 1 && currentSourceUpperIndex >= 2)) {
                        inSourceBlock = cellLonIndex >= src_lon && cellLonIndex < src_lon + patchSize &&
                                        (cellLatIndex % blockHeight) >= src_lat && (cellLatIndex % blockHeight) < src_lat + patchSize;
                    }
                }

                if (inTargetBlock) {
                    return mixColors(originalColor, "pink", 0.4);
                } else if (inSourceBlock) {
                    return mixColors(originalColor, "cyan", 0.4);
                } else {
                    return originalColor;
                }
            });
    }

    // Attach event listeners
    function attachEventListeners() {
        const eventListeners = {
            'variable-select': (value) => { state.currentSurfaceVariable = value; },
            'upper-variable-select': (value) => { state.currentUpperVariable = value; },
            'attention-head': (value) => { state.currentAttentionHead = parseInt(value, 10); },
            'latitude': (value) => { state.currentLatIndex = parseInt(value, 10); },
            'longitude': (value) => { state.currentLonIndex = parseInt(value, 10); },
            'pressure_level': (value) => { state.currentPressureLevel = parseInt(value, 10); },
            'date-select': (value) => { state.currentDate = value; populateTimeAndLayerSelects(); },
            'time-select': (value) => { state.currentTime = value; populateTimeAndLayerSelects(); },
            'layer-select': (value) => { state.currentLayer = value; updateAttentionHeadOptions(); }
        };

        Object.keys(eventListeners).forEach(id => {
            document.getElementById(id).addEventListener('change', (event) => {
                eventListeners[id](event.target.value);
                initializeVisualizations();
            });
        });
    }

    // Initialize visualizations on load
    attachEventListeners();
    initializeVisualizations();
});
