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

// Function to fetch and parse JSON data with error handling
function fetchAndParseJSON(url) {
    return fetch(url)
        .then(response => response.json())
        .catch(error => {
            console.error(`Error fetching or parsing JSON from ${url}:`, error);
            throw error;
        });
}

// Function to load a specific map chunk
function loadMapChunk(latIndex, lonIndex, variable) {
    const lat = latIndex * 24;
    const lon = lonIndex * 48;
    const url = `json/input_surface_${lat}_${lon}.json`;
    return fetchAndParseJSON(url).then(data => {
        return data[surfaceVarIdx[variable]];
    });
}

// Function to load a specific attention window
function loadAttentionWindow(winLon, winLatPl, head) {
    const url = `json/attention_${winLatPl}_${winLon}_${head}.json`;
    return fetchAndParseJSON(url);
}

// Initialize visualizations with default chunks
function initializeVisualizations() {
    Promise.all([
        loadMapChunk(currentLatIndex, currentLonIndex, currentVariable),
        loadAttentionWindow(currentLonIndex, currentLatIndex, currentAttentionHead)
    ]).then(([mapData, attentionData]) => {
        initMap(mapData);
        initAttentionPattern(attentionData, mapData);
    }).catch(error => {
        console.error('Failed to initialize visualizations:', error);
    });
}

// Add event listeners for variable selection
document.getElementById('variable-select').addEventListener('change', (event) => {
    currentVariable = event.target.value;  // Update the global state
    loadMapChunk(currentLatIndex, currentLonIndex, currentVariable).then(updateMap);
});

// Add event listeners for attention head selection
document.getElementById('attention-head').addEventListener('change', (event) => {
    currentAttentionHead = parseInt(event.target.value, 10);  // Update the global state
    loadAttentionWindow(currentLonIndex, currentLatIndex, currentAttentionHead).then(updateAttentionPattern);
});

// Add event listeners for latitude and longitude index inputs
document.getElementById('latitude').addEventListener('change', (event) => {
    currentLatIndex = parseInt(event.target.value, 10);  // Update the global state
    updateVisualizations();
});

document.getElementById('longitude').addEventListener('change', (event) => {
    currentLonIndex = parseInt(event.target.value, 10);  // Update the global state
    updateVisualizations();
});

// Function to update visualizations when indices change
function updateVisualizations() {
    Promise.all([
        loadMapChunk(currentLatIndex, currentLonIndex, currentVariable),
        loadAttentionWindow(currentLonIndex, currentLatIndex, currentAttentionHead)
    ]).then(([mapData, attentionData]) => {
        updateMap(mapData);
        updateAttentionPattern(attentionData);
    }).catch(error => {
        console.error('Failed to update visualizations:', error);
    });
}

function initMap(data) {
    const svg = d3.select("#map-container").append("svg")
        .attr("viewBox", "0 0 720 720")
        .attr("preserveAspectRatio", "xMidYMid meet")
        .classed("svg-content-responsive", true);

    const colorScale = d3.scaleSequential(d3.interpolateTurbo)
        .domain(d3.extent(data.flat()));

    svg.selectAll("rect")
        .data(data.flat())
        .enter().append("rect")
        .attr("x", (d, i) => (i % 48) * 15)
        .attr("y", (d, i) => Math.floor(i / 48) * 15)
        .attr("width", 15)
        .attr("height", 15)
        .attr("fill", d => colorScale(d))
        .attr("class", "map-cell")
        .attr("data-original-color", d => colorScale(d));
}

function updateMap(data) {
    const svg = d3.select("#map-container svg");

    const colorScale = d3.scaleSequential(d3.interpolateTurbo)
        .domain(d3.extent(data.flat()));

    svg.selectAll("rect.map-cell")
        .data(data.flat())
        .transition()
        .duration(500)
        .attr("fill", d => colorScale(d))
        .attr("data-original-color", d => colorScale(d));
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

function updateAttentionPattern(attentionData) {
    const svg = d3.select("#attention-container svg");

    const colorScale = d3.scaleSequential(d3.interpolateViridis)
        .domain(d3.extent(attentionData.flat()));

    svg.selectAll("rect")
        .data(attentionData.flat())
        .transition()
        .duration(500)
        .attr("fill", d => colorScale(d));
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

// Initialize the visualizations on page load
initializeVisualizations();
