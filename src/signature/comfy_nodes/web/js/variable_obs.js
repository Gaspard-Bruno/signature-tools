import { app } from "../../../scripts/app.js";

const NODES = {
    "signature_input_image": "Input Image",
    "signature_input_text": "Input Text",
    "signature_input_number": "Input Number",
    "signature_output": "Output",
};

const COLOR_THEMES = {
    red: { nodeColor: "#332222", nodeBgColor: "#553333" },
    green: { nodeColor: "#223322", nodeBgColor: "#335533" },
    blue: { nodeColor: "#222233", nodeBgColor: "#333355" },
    pale_blue: { nodeColor: "#2a363b", nodeBgColor: "#3f5159" },
    cyan: { nodeColor: "#223333", nodeBgColor: "#335555" },
    purple: { nodeColor: "#332233", nodeBgColor: "#553355" },
    yellow: { nodeColor: "#443322", nodeBgColor: "#665533" },
    none: { nodeColor: null, nodeBgColor: null } // no color
};

function setNodeColors(node, theme) {
    if (!theme) {return;}
    node.shape = "box";
    if(theme.nodeColor && theme.nodeBgColor) {
        node.color = theme.nodeColor;
        node.bgcolor = theme.nodeBgColor;
    }
}

function output(node, widget) {
    setNodeColors(node, COLOR_THEMES['purple']);
    const widgetType = widget.value.toUpperCase();
    if (node.inputs !== undefined) {
        for (const input of node.inputs) {
            if (input.name === "value") {
                input.type = widgetType;
            }
        }
    }

    const widgets = node.widgets || []
    for (const w of widgets) {
        if (w.name === "title") {
            if (w.value.startsWith("Output ")) {
                const modStr = widget.value[0].toUpperCase() + widget.value.slice(1);
                w.value = "Output " + modStr;
            }
            break;
        }
    }
}

function inputImage(node, widget) {
    setNodeColors(node, COLOR_THEMES['pale_blue']);
    const name = widget.value;
    const type = widget.value.toUpperCase();
    if (node.inputs !== undefined) {
        node.inputs[0].type = type;
    } else {
        node.addInput(name, type);
    }

    if (node.outputs !== undefined) {
        node.outputs[0].name = name;
        node.outputs[0].type = type;
    } else {
        node.addOutput(name, type);
    }
}

function inputText(node, widget) {
    const value = widget.value;
    if (value === 'string') {
        setNodeColors(node, COLOR_THEMES['yellow']);
    }

    if (value === 'positive_prompt') {
        setNodeColors(node, COLOR_THEMES['green']);
    }

    if (value === 'negative_prompt') {
        setNodeColors(node, COLOR_THEMES['red']);
    }
}

function getNumberDefaults(inputData, defaultStep, precision, enable_rounding) {
	let defaultVal = inputData[1]["default"];
	let { min, max, step, round} = inputData[1];

	if (defaultVal == undefined) defaultVal = 0;
	if (min == undefined) min = 0;
	if (max == undefined) max = 2048;
	if (step == undefined) step = defaultStep;
	// precision is the number of decimal places to show.
	// by default, display the the smallest number of decimal places such that changes of size step are visible.
	if (precision == undefined) {
		precision = Math.max(-Math.floor(Math.log10(step)),0);
	}

	if (enable_rounding && (round == undefined || round === true)) {
		// by default, round the value to those decimal places shown.
		round = Math.round(1000000*Math.pow(0.1,precision))/1000000;
	}

	return { val: defaultVal, config: { min, max, step: 10.0 * step, round, precision } };
}

function inputNumber(node, widget) {
    setNodeColors(node, COLOR_THEMES['cyan']);

    const widgetType = widget.value.toUpperCase();
    if (node.inputs !== undefined) {
        if (node.inputs.length > 0) {
            if (node.inputs[0].name === 'value') {
                node.inputs[0].type = widgetType;
            }
        }
    }
    if (node.outputs !== undefined) {
        node.outputs[0].type = widgetType;
        node.outputs[0].name = widget.value;
    }

    const widgets = node.widgets || []
    let valueWidget = null;
    for (const w of widgets) {
        if (w.name === "value") {
            valueWidget = w;
            break;
        }
    }

    if (valueWidget !== null) {
        if (widget.value === "int") {
            valueWidget.options.precision = 0;
            valueWidget.options.round = 0;
            valueWidget.options.step = 1;
        } else {
            valueWidget.options.precision = 2;
            valueWidget.options.round = 0.01;
            valueWidget.options.step = 0.01;
        }
    }
}


const nodeWidgetHandlers = {
    "signature_input_image": {
        'subtype': inputImage
    },
    "signature_input_text": {
        'subtype': inputText
    },
    "signature_input_number": {
        'subtype': inputNumber
    },
    "signature_output": {
        'subtype': output
    },
};

 // In the main function where widgetLogic is called
function widgetLogic(node, widget) {
    // Retrieve the handler for the current node title and widget name
    const handler = nodeWidgetHandlers[node.comfyClass]?.[widget.name];
    if (handler) {
        handler(node, widget);
    }
}

const ext = {
    name: "signature.platform_io",

    nodeCreated(node) {
        const title = node.getTitle();
        if (NODES.hasOwnProperty(title)) {
            node.title = NODES[title];

            node.validateLinks = function() {
                if (node.outputs !== undefined) {
                    if (node.outputs[0].type !== '*' && node.outputs[0].links) {
                        node.outputs[0].links.filter(linkId => {
                            const link = node.graph.links[linkId];
                            return link && (link.type !== node.outputs[0].type && link.type !== '*');
                        }).forEach(linkId => {
                            node.graph.removeLink(linkId);
                        });
                    }
                }
            };


            // node.update = function() {

            //     if (node.graph === undefined) {
            //         return;
            //     }

            //     if (!node.graph) {
            //         return;
            //     }

            //     if (node.inputs !== undefined) {
            //         const getters = graph._nodes.filter(x => x.type === node.type);
            //         getters.forEach(getter => {
            //             if (getter !== undefined && getter.inputs !== undefined) {
            //                 const nodeType = node.inputs[0].type;
            //                 getter.outputs[0].type = nodeType;
            //                 getter.outputs[0].name = nodeType;
            //                 getter.validateLinks();
            //             }
            //         });
            //     }

            // }

            for (const w of node.widgets || []) {
                let widgetValue = w.value;
                widgetLogic(node, w);
                // Store the original descriptor if it exists
                let originalDescriptor = Object.getOwnPropertyDescriptor(w, 'value');

                Object.defineProperty(w, 'value', {
                    get() {
                        // If there's an original getter, use it. Otherwise, return widgetValue.
                        let valueToReturn = originalDescriptor && originalDescriptor.get
                            ? originalDescriptor.get.call(w)
                            : widgetValue;
                        return valueToReturn;
                    },
                    set(newVal) {
                        // If there's an original setter, use it. Otherwise, set widgetValue.
                        if (originalDescriptor && originalDescriptor.set) {
                            originalDescriptor.set.call(w, newVal);
                        } else {
                            widgetValue = newVal;
                        }

                        widgetLogic(node, w);
                        // node.update();
                    }
                });
            }
        }
    }
};

app.registerExtension(ext);