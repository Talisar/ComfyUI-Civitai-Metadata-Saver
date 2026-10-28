import { app } from "../../scripts/app.js";

const NODES = ["LoadTextByIndex", "LoadImageByIndex"];

// ComfyUI starts every control_after_generate widget on "randomize". These nodes are meant to
// step through a folder, so new copies start on "increment". Saved workflows keep their own value.
app.registerExtension({
  name: "Talisar.FolderLoaders",
  nodeCreated(node) {
    if (!NODES.includes(node.comfyClass)) return;
    const control = node.widgets?.find((w) => w.name === "control_after_generate");
    if (control) control.value = "increment";
  },
});
