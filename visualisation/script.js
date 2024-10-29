document.addEventListener('DOMContentLoaded', () => {
    // Define the mapping of weather variables to their indices
    const surfaceVarIdx = {
        "MSLP": 0,
        "U10": 1,
        "V10": 2,
        "T2M": 3
    };

    // Global state to keep track of the selected variable, attention head, and indices
    let currentVariable = 'T2M';
    let currentAttentionHead = 0;
    let currentLatIndex = 0;
    let currentLonIndex = 0;

    // Date and time for data organization
    const dataDate = "2018-01-01";
    const dataTime = "00:00";

    // Function to load binary data and convert it to a tensor
    async function loadBinaryData(url, shape, dtype = 'float32') {
        const response = await fetch(url);
        const arrayBuffer = await response.arrayBuffer();
        const typedArray = new Float32Array(arrayBuffer);
        if (typedArray.length !== shape.reduce((a, b) => a * b)) {
            throw new Error(`Mismatch in data size: expected ${shape.reduce((a, b) => a * b)} but got ${typedArray.length}`);
        }
        return tf.tensor(typedArray, shape, dtype);
    }

    // Function to load a specific map chunk
    async function loadMapChunk(latIndex, lonIndex) {
        const url = `bin/${dataDate}/${dataTime}/map/input_surface_${latIndex * 24}_${lonIndex * 48}.bin`;
        return loadBinaryData(url, [4, 24, 48]);
    }

    // Function to load a specific attention chunk
    async function loadAttentionChunk(lon, latPl, head) {
        const url = `bin/${dataDate}/${dataTime}/attention/attention_${lon}_${latPl}_${head}.bin`;
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

            initMap(mapDataArray.flat());
            initAttentionPattern(attentionDataArray, mapDataArray);
        } catch (error) {
            console.error('Failed to initialize visualizations:', error);
        }
    }

    function initMap(data) {
        const svg = d3.select("#map-container").append("svg")
            .attr("viewBox", "0 0 720 720")
            .attr("preserveAspectRatio", "xMidYMid meet")
            .classed("svg-content-responsive", true);

        const dataExtent = d3.extent(data.flat());
        console.log("Data Extent:", dataExtent); // Debugging line

        if (dataExtent[0] === dataExtent[1]) {
            console.warn("Data extent has no range, all values might be the same.");
        }

        const colorScale = d3.scaleSequential(d3.interpolateTurbo)
            .domain(dataExtent);

        svg.selectAll("rect")
            .data(data.flat())
            .enter().append("rect")
            .attr("x", (d, i) => (i % 48) * 15)
            .attr("y", (d, i) => Math.floor(i / 48) * 15)
            .attr("width", 15)
            .attr("height", 15)
            .attr("fill", d => {
                if (d == null || isNaN(d)) {
                    return "gray";
                }
                return colorScale(d);
            })
            .attr("class", "map-cell")
            .attr("data-original-color", d => {
                if (d == null || isNaN(d)) {
                    return "gray";
                }
                return colorScale(d);
            });
    }

    function initAttentionPattern(attentionData, mapData) {
        const svg = d3.select("#attention-container").append("svg")
            .attr("viewBox", "0 0 864 864")
            .attr("preserveAspectRatio", "xMidYMid meet")
            .classed("svg-content-responsive", true);

        const colorScale = d3.scaleSequential(d3.interpolateViridis)
            .domain(d3.extent(attentionData.flat()));

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
        const x = (attentionIndex % 144);
        const y = Math.floor(attentionIndex / 144);

        const tar_pl = Math.floor(x / (12 * 6));
        const src_pl = Math.floor(y / (12 * 6));

        const tar_lat = (Math.floor((x) / 12) % 6) * 4;
        const src_lat = (Math.floor((y) / 12) % 6) * 4;

        const tar_lon = (x % 12) * 4;
        const src_lon = (y % 12) * 4;

        const tar_index = tar_lon + tar_lat * 48;
        const src_index = src_lon + src_lat * 48;

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

                const tar_x = tar_index % 48;
                const tar_y = Math.floor(tar_index / 48);
                const src_x = src_index % 48;
                const src_y = Math.floor(src_index / 48);

                const inTargetBlock = (i % 48 >= tar_x && i % 48 < tar_x + 4) &&
                    (Math.floor(i / 48) >= tar_y && Math.floor(i / 48) < tar_y + 4);

                const inSourceBlock = (i % 48 >= src_x && i % 48 < src_x + 4) &&
                    (Math.floor(i / 48) >= src_y && Math.floor(i / 48) < src_y + 4);

                if (inTargetBlock) {
                    return mixColors(originalColor, "pink", 0.35);
                } else if (inSourceBlock) {
                    return mixColors(originalColor, "cyan", 0.35);
                } else {
                    return originalColor;
                }
            });
    }

    // Event listeners for user input
    document.getElementById('variable-select').addEventListener('change', (event) => {
        currentVariable = event.target.value;
        initializeVisualizations();
    });

    document.getElementById('attention-head').addEventListener('change', (event) => {
        currentAttentionHead = parseInt(event.target.value, 10);
        initializeVisualizations();
    });

    document.getElementById('latitude').addEventListener('input', (event) => {
        currentLatIndex = parseInt(event.target.value, 10);
        initializeVisualizations();
    });

    document.getElementById('longitude').addEventListener('input', (event) => {
        currentLonIndex = parseInt(event.target.value, 10);
        initializeVisualizations();
    });

    // Initialize the visualizations on page load
    initializeVisualizations();
});
