document.addEventListener('DOMContentLoaded', async () => {
    const surfaceVarIdx = {
        "MSLP": 0,
        "U10": 1,
        "V10": 2,
        "T2M": 3
    };

    const intermediateLayerNames = [
        '/b1/Add_output_0', '/b1/Add_3_output_0', '/b1/Add_7_output_0', '/b1/Add_10_output_0',
        '/b1/Add_14_output_0', '/b1/Add_17_output_0', '/b1/Add_21_output_0', '/b1/Add_24_output_0',
        '/b1/Add_28_output_0', '/b1/Add_31_output_0', '/b1/Add_35_output_0', '/b1/Add_38_output_0',
        '/b1/Add_42_output_0', '/b1/Add_45_output_0', '/b1/Add_49_output_0', '/b1/Add_52_output_0',
    ];

    let currentVariable = 'T2M';
    let currentAttentionHead = 0;
    let currentLatIndex = 0;
    let currentLonIndex = 0;
    let currentDate = "";
    let currentTime = "";
    let currentLayer = "";

    const availableData = await fetch('available_data.json').then(response => response.json());

    const dateSelect = document.getElementById('date-select');
    const timeSelect = document.getElementById('time-select');
    const layerSelect = document.getElementById('layer-select');
    const attentionHeadSelect = document.getElementById('attention-head');

    Object.keys(availableData).forEach(date => {
        const option = new Option(date, date);
        dateSelect.add(option);
    });
    currentDate = dateSelect.value;

    function populateTimeAndLayerSelects() {
        timeSelect.innerHTML = '';
        layerSelect.innerHTML = '';

        Object.keys(availableData[currentDate]).forEach(time => {
            const option = new Option(time, time);
            timeSelect.add(option);
        });
        currentTime = timeSelect.value;

        availableData[currentDate][currentTime].forEach(layer => {
            const option = new Option(layer, layer);
            layerSelect.add(option);
        });
        currentLayer = layerSelect.value;
        updateAttentionHeadOptions();
    }

    function updateAttentionHeadOptions() {
        attentionHeadSelect.innerHTML = '';
        const { numHeads } = getLayerConfig(currentLayer);
        for (let i = 0; i < numHeads; i++) {
            const option = new Option(`Head ${i}`, i);
            attentionHeadSelect.add(option);
        }
        currentAttentionHead = attentionHeadSelect.value;
    }

    populateTimeAndLayerSelects();

    function getLayerConfig(layerName) {
        const modifiedLayerName = layerName.replace(/^_/, '/').replace('_', '/');
        const layerIndex = intermediateLayerNames.indexOf(modifiedLayerName);
        return (layerIndex < 2 || layerIndex >= intermediateLayerNames.length - 2) ?
            { numHeads: 6, configName: 'config_24x48', chunkSize: [24, 48] } :
            { numHeads: 12, configName: 'config_48x96', chunkSize: [48, 96] };
    }

    async function loadBinaryData(url, shape, dtype = 'float32') {
        const response = await fetch(url);
        const arrayBuffer = await response.arrayBuffer();
        const typedArray = new Float32Array(arrayBuffer);
        if (typedArray.length !== shape.reduce((a, b) => a * b)) {
            throw new Error(`Mismatch in data size: expected ${shape.reduce((a, b) => a * b)} but got ${typedArray.length}`);
        }
        return tf.tensor(typedArray, shape, dtype);
    }

    async function loadMapChunk(latIndex, lonIndex) {
        const { configName, chunkSize } = getLayerConfig(currentLayer);
        const url = `bin/${currentDate}/${currentTime}/${configName}/map/input_surface_${latIndex * chunkSize[0]}_${lonIndex * chunkSize[1]}.bin`;
        return loadBinaryData(url, [4, chunkSize[0], chunkSize[1]]);
    }

    async function loadAttentionChunk(lon, latPl, head) {
        const { numHeads } = getLayerConfig(currentLayer);
        if (head >= numHeads) {
            throw new Error(`Invalid head index: ${head} for layer ${currentLayer}`);
        }
        const url = `bin/${currentDate}/${currentTime}/${currentLayer}/attention/attention_${lon}_${latPl}_${head}.bin`;
        return loadBinaryData(url, [144, 144]);
    }

    async function initializeVisualizations() {
        try {
            const mapData = await loadMapChunk(currentLatIndex, currentLonIndex);
            const attentionData = await loadAttentionChunk(currentLonIndex, currentLatIndex, currentAttentionHead);

            const mapDataArray = mapData.gather([surfaceVarIdx[currentVariable]], 0).arraySync();
            const attentionDataArray = attentionData.arraySync();

            d3.select("#map-container").selectAll("*").remove();
            d3.select("#attention-container").selectAll("*").remove();

            initMap(mapDataArray.flat(), mapData.shape);
            initAttentionPattern(attentionDataArray, mapDataArray);
        } catch (error) {
            console.error('Failed to initialize visualizations:', error);
        }
    }

    function initMap(data, mapShape) {
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

    function highlightMapCells(attentionIndex, mapData) {
        const { chunkSize } = getLayerConfig(currentLayer);
        const mapWidth = chunkSize[1];
        
        if (mapWidth == 48) {
            patchSize = 4;
        } else {
            patchSize = 8;
        }

        const x = (attentionIndex % 144);
        const y = Math.floor(attentionIndex / 144);

        const tar_pl = Math.floor(x / (12 * 6));
        const src_pl = Math.floor(y / (12 * 6));

        const tar_lat = (Math.floor(x / 12) % 6) * patchSize;
        const src_lat = (Math.floor(y / 12) % 6) * patchSize;

        const tar_lon = (x % 12) * patchSize;
        const src_lon = (y % 12) * patchSize;

        const tar_index = tar_lon + tar_lat * mapWidth;
        const src_index = src_lon + src_lat * mapWidth;

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

                const tar_x = tar_index % mapWidth;
                const tar_y = Math.floor(tar_index / mapWidth);
                const src_x = src_index % mapWidth;
                const src_y = Math.floor(src_index / mapWidth);

                const inTargetBlock = (i % mapWidth >= tar_x && i % mapWidth < tar_x + patchSize) &&
                    (Math.floor(i / mapWidth) >= tar_y && Math.floor(i / mapWidth) < tar_y + patchSize);

                const inSourceBlock = (i % mapWidth >= src_x && i % mapWidth < src_x + patchSize) &&
                    (Math.floor(i / mapWidth) >= src_y && Math.floor(i / mapWidth) < src_y + patchSize);

                if (inTargetBlock) {
                    return mixColors(originalColor, "pink", 0.4);
                } else if (inSourceBlock) {
                    return mixColors(originalColor, "cyan", 0.4);
                } else {
                    return originalColor;
                }
            });
    }

    const eventListeners = {
        'variable-select': (event) => {
            currentVariable = event.target.value;
            initializeVisualizations();
        },
        'attention-head': (event) => {
            currentAttentionHead = parseInt(event.target.value, 10);
            initializeVisualizations();
        },
        'latitude': (event) => {
            currentLatIndex = parseInt(event.target.value, 10);
            initializeVisualizations();
        },
        'longitude': (event) => {
            currentLonIndex = parseInt(event.target.value, 10);
            initializeVisualizations();
        },
        'date-select': (event) => {
            currentDate = event.target.value;
            populateTimeAndLayerSelects();
            initializeVisualizations();
        },
        'time-select': (event) => {
            currentTime = event.target.value;
            populateTimeAndLayerSelects();
            initializeVisualizations();
        },
        'layer-select': (event) => {
            currentLayer = event.target.value;
            updateAttentionHeadOptions();
            initializeVisualizations();
        }
    };

    Object.keys(eventListeners).forEach(id => {
        document.getElementById(id).addEventListener('change', eventListeners[id]);
    });

    initializeVisualizations();
});
