import os
import webbrowser
import ase.io
import homcloud.interface as hc
import homcloud.py3dmolhelper as py3dmolhelper
import numpy as np
import py3Dmol
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

xyz_file = "input.xyz"

epsilon = 0.0005

ring_color = "green"
ring_radius = 0.2

atom_opacity = 0.35
atom_scale = 0.3

show_axes = True
axis_length = 5.0
axis_radius = 0.12

amorph_ice = ase.io.read(xyz_file)
weights = np.array([0.175**2 if atom == "O" else 0.775**2 for atom in amorph_ice.get_chemical_symbols()])

pd1 = hc.PDList.from_alpha_filtration(
    amorph_ice.get_positions(),
    vertex_symbols=amorph_ice.get_chemical_symbols(),
    weight=weights,
    save_boundary_map=True,
    save_phtrees=True,
    save_to="NAME.pdgm"
).dth_diagram(1)

pd1.histogram((0, 6), 250).plot(colorbar={"type": "log", "colormap": "plasma"}, font_size=18)
plt.show()


#HDA_maskcoords_H1 = [(1.3, 1.75),(1, 1.25),(0.75, 1),(0.6, 1.25),(0.75, 1.75),(1, 2)] #full molecule
HDA_maskcoords_H1 = [(2.75, 3.5),(2.25, 3.5),(1.75, 3),(1.75, 2.5)] #Ocentric

polygon = Polygon(HDA_maskcoords_H1)

relevant_pairs = [pair for pair in pd1.pairs()if polygon.contains(Point(pair.birth, pair.death))]

print(f"Selected {len(relevant_pairs)} persistent pairs")

stable_volumes = []
for i, pair in enumerate(relevant_pairs):
    try:
        stable_volume = pair.stable_volume(epsilon)
        stable_volumes.append(stable_volume)

        print(
            f"Ring {i + 1}: "
            f"birth={pair.birth:.6f}, "
            f"death={pair.death:.6f}"
        )

    except Exception as exc:
        print(f"Ring {i + 1} failed: {exc}")

print(f"Successfully extracted {len(stable_volumes)} rings")

with open(xyz_file, "r") as f:
    amorphous_ice_xyz = f.read()

view = py3Dmol.view(width=1000, height=750)
view.addModel(amorphous_ice_xyz, "xyz")
view.setStyle({"stick": {"colorscheme": "default","opacity": atom_opacity}})
view.addStyle({"sphere": {"scale": atom_scale,"opacity": atom_opacity}})

for stable_volume in stable_volumes:
    py3dmolhelper.add_edges(view,stable_volume.boundary(),ring_color,ring_radius)

if show_axes:
    origin = amorph_ice.get_positions().min(axis=0)
    axes = [
        ("X", [1, 0, 0], "red"),
        ("Y", [0, 1, 0], "green"),
        ("Z", [0, 0, 1], "blue")]

    for label, direction, color in axes:
        endpoint = origin + axis_length * np.array(direction)
        start = dict(zip(("x", "y", "z"), map(float, origin)))
        end = dict(zip(("x", "y", "z"), map(float, endpoint)))

        view.addArrow({"start": start,"end": end,"color": color,"radius": axis_radius})
        view.addLabel(label, {"position": end,"fontColor": color,"backgroundOpacity": 0,"fontSize": 18})


view.setBackgroundColor("white")
view.zoomTo()
view.setBackgroundColor("white")
view.zoomTo()

html = view._make_html()
viewer_id = f"viewer_{view.uniqueid}"
container_id = f"3dmolviewer_{view.uniqueid}"

render_call = f"{viewer_id}.render();" #give py3dmol viewer my controls

if render_call not in html:
    raise RuntimeError("Could not locate the 3Dmol viewer.")

html = html.replace(render_call,render_call + f"""window.ringViewer = {viewer_id};window.ringBaseView = {viewer_id}.getView().slice();""",1)

controls = """
<div style="font-family:Arial; padding:15px;">

    <h3>Ring Visualization Controls</h3>

    <div>
        X rotation:
        <input type="range" id="rotX"
               min="-180" max="180" value="0"
               oninput="updateRotation()">
        <span id="valX">0</span>°
    </div>

    <div>
        Y rotation:
        <input type="range" id="rotY"
               min="-180" max="180" value="0"
               oninput="updateRotation()">
        <span id="valY">0</span>°
    </div>

    <div>
        Z rotation:
        <input type="range" id="rotZ"
               min="-180" max="180" value="0"
               oninput="updateRotation()">
        <span id="valZ">0</span>°
    </div>

    <p>
        <button onclick="resetRotation()">
            Reset rotation
        </button>
    </p>

    <hr>

    <label>PNG resolution:</label>

    <select id="resolution">
        <option value="1">1000 x 750</option>
        <option value="2">2000 x 1500</option>
        <option value="3">3000 x 2250</option>
    </select>

    <button onclick="savePNG()">Save PNG</button>

    <p id="status"></p>

</div>
"""

javascript = r"""
<script>

function getViewer() {
    if (!window.ringViewer) {
        document.getElementById("status").textContent =
            "3D viewer has not finished loading.";
        return null;
    }
    return window.ringViewer;
}

function updateRotation() {

    const viewer = getViewer();
    if (!viewer) return;

    const x = Number(document.getElementById("rotX").value);
    const y = Number(document.getElementById("rotY").value);
    const z = Number(document.getElementById("rotZ").value);

    document.getElementById("valX").textContent = x;
    document.getElementById("valY").textContent = y;
    document.getElementById("valZ").textContent = z;

    // Restore initial orientation
    viewer.setView(window.ringBaseView.slice());

    // Apply rotations in X, Y, Z order
    viewer.rotate(x, "x");
    viewer.rotate(y, "y");
    viewer.rotate(z, "z");

    viewer.render();
}

function resetRotation() {

    for (const axis of ["X", "Y", "Z"]) {
        document.getElementById("rot" + axis).value = 0;
    }

    updateRotation();
}

function savePNG() {

    const viewer = getViewer();
    if (!viewer) return;

    const container = document.getElementById(
        "VIEWER_CONTAINER_ID"
    );

    const scale = Number(
        document.getElementById("resolution").value
    );

    const oldWidth = container.style.width;
    const oldHeight = container.style.height;
    const oldView = viewer.getView().slice();

    const status = document.getElementById("status");

    try {
        // Increase canvas resolution
        container.style.width = (1000 * scale) + "px";
        container.style.height = (750 * scale) + "px";

        viewer.resize();
        viewer.setView(oldView);
        viewer.render();

        requestAnimationFrame(function() {

            try {
                const png = viewer.pngURI();

                const link = document.createElement("a");
                link.href = png;
                link.download = "ring_projection.png";

                document.body.appendChild(link);
                link.click();
                link.remove();

                status.textContent = "PNG export requested.";

            } catch (error) {
                status.textContent = "Export failed: " + error.message;

            } finally {
                // Restore original viewer dimensions
                container.style.width = oldWidth;
                container.style.height = oldHeight;

                viewer.resize();
                viewer.setView(oldView);
                viewer.render();
            }
        });

    } catch (error) {
        container.style.width = oldWidth;
        container.style.height = oldHeight;
        viewer.resize();
        viewer.setView(oldView);
        viewer.render();
        status.textContent = "Export failed: " + error.message;
    }
}

</script>
"""

javascript = javascript.replace(
    "VIEWER_CONTAINER_ID",
    container_id
)


full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Ring Visualization</title>
</head>

<body>

{controls}

{html}

{javascript}

</body>
</html>
"""

html_file = os.path.abspath("rings_visualization.html")

with open(html_file, "w", encoding="utf-8") as f:
    f.write(full_html)

webbrowser.open("file://" + html_file)
