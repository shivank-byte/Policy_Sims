"""
streamlit_app.py
-----------------
Single-file entrypoint for Streamlit Cloud. This is the ONE thing you point
Streamlit Cloud at — no separate FastAPI server needed.

- The sidebar holds real controls (Step round, Restart, 4 policy toggles)
  that call straight into the actual Simulation object in st.session_state.
- The main panel embeds the 3D Three.js village via components.html, fed
  the current simulation snapshot as embedded JSON. Clicking a building
  inside the 3D view (households/firms/town hall) opens a read-only
  inspector panel — that part is pure client-side JS, no round-trip needed.
- Because Streamlit reruns the whole script on every widget click, the
  iframe is rebuilt each time too, so the camera angle resets on each
  step/toggle. That's a known trade-off of embedding a 3D view this way;
  drag to re-orbit after each click.

Deploy: point Streamlit Cloud at this file (repo root), main file path
`streamlit_app.py`.
"""

import json
import io
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

from policysim.simulation import Simulation
from policysim.brain import AgentBrain
from policysim.policies import POLICY_LIBRARY, COLOR_TO_POLICY
from policysim.view_helpers import snapshot

try:
    from policysim.trigger_system import ColorCardDetector, QRTriggerDetector
    _CAMERA_TRIGGER_AVAILABLE = True
    _CAMERA_IMPORT_ERROR = None
except Exception as _e:  # e.g. opencv failed to import on this host
    ColorCardDetector = None
    QRTriggerDetector = None
    _CAMERA_TRIGGER_AVAILABLE = False
    _CAMERA_IMPORT_ERROR = str(_e)

st.set_page_config(page_title="PolicySim — Econ Village 3D", layout="wide")

POLICY_COLOR = {
    "subsidy_cut": "#e8604c",
    "minimum_wage_increase": "#c9a227",
    "luxury_tax": "#4c72a8",
    "cash_transfer": "#6fae7a",
}

# ---------------------------------------------------------------- state ---

def _make_simulation() -> Simulation:
    """Uses the Groq backend if GROQ_API_KEY is set in Streamlit secrets,
    otherwise falls back to the zero-dependency heuristic backend. Never
    put the key in code — set it via Settings → Secrets on Streamlit Cloud:
        GROQ_API_KEY = "your-key-here"
    """
    groq_key = st.secrets.get("GROQ_API_KEY", None)
    if groq_key:
        brain = AgentBrain(backend="groq", groq_api_key=groq_key)
    else:
        brain = AgentBrain(backend="heuristic")
    return Simulation(brain=brain)


if "sim" not in st.session_state:
    st.session_state.sim = _make_simulation()

sim = st.session_state.sim

# -------------------------------------------------------------- sidebar ---

with st.sidebar:
    st.markdown("## PolicySim")
    backend_label = "🧠 Groq (live LLM reasoning)" if sim.brain.backend == "groq" else "⚙️ Heuristic (rule-based)"
    st.caption(f"Live engine · 10 households · 2 firms · 1 government")
    st.caption(f"Backend: **{backend_label}**")
    if sim.brain.backend == "groq" and sim.brain.fallback_count > 0:
        st.caption(f"⚠️ Groq failed {sim.brain.fallback_count}x this session — auto-fell back to heuristic on those calls.")

    st.metric("Round", sim.round_num)

    col1, col2 = st.columns(2)
    if col1.button("Step round →", use_container_width=True):
        sim.run_round()
    if col2.button("Restart", use_container_width=True):
        st.session_state.sim = _make_simulation()
        sim = st.session_state.sim

    if sim.history:
        latest = sim.history[-1]
        m1, m2, m3 = st.columns(3)
        m1.metric("Price idx", f"{latest['price_level']:.2f}")
        m2.metric("Unemployment", f"{latest['unemployment_rate']*100:.0f}%")
        m3.metric("Wealth Gini", f"{latest['gini']:.2f}")

    st.markdown("---")
    st.markdown("### Policy levers")
    st.caption("Tap to trigger a shock. Tap again to revert it.")

    for policy_id, spec in POLICY_LIBRARY.items():
        active = policy_id in sim.government.active_policies
        label = f"{'🟢' if active else '⚪'} {spec['label']}"
        if st.button(label, key=f"policy_{policy_id}", use_container_width=True, help=spec["description"]):
            sim.apply_policy_event(policy_id)

    st.markdown("---")
    st.markdown("### 📷 Or trigger with a camera")
    if not _CAMERA_TRIGGER_AVAILABLE:
        st.caption(
            f"Camera trigger unavailable on this deployment (opencv failed to import: "
            f"{_CAMERA_IMPORT_ERROR}). If deploying to Streamlit Cloud, add a `packages.txt` "
            "with `libgl1` and `libglib2.0-0` -- see README §0.1."
        )
    else:
        st.caption(
            "Uses your device's own camera through the browser (not a server-side webcam -- "
            "this works the same on a phone as on a laptop, including when this app is hosted "
            "on Streamlit Cloud). Point at a color card or QR trigger card and capture."
        )

        def _handle_trigger_photo(raw_bytes: bytes):
            rgb_image = np.array(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))
            bgr_frame = rgb_image[:, :, ::-1]  # PIL gives RGB, OpenCV expects BGR
            policy_id, magnitude = QRTriggerDetector().decode_policy(bgr_frame)  # QR checked first (carries magnitude)
            detected_color = None
            if not policy_id:
                detected_color = ColorCardDetector().detect(bgr_frame)
                policy_id = COLOR_TO_POLICY.get(detected_color) if detected_color else None
            if policy_id and policy_id in POLICY_LIBRARY:
                msg = sim.apply_policy_event(policy_id, magnitude)
                label = POLICY_LIBRARY[policy_id]["label"]
                via = f"**{detected_color}** card" if detected_color else "QR code"
                st.success(f"Detected {via} → **{label}**. {msg}")
            else:
                st.warning(
                    "No recognized card or QR code found. Try: fill more of the frame with "
                    "the card, avoid backlighting/glare, and use a plain matte red, yellow, "
                    "blue, or green card. If color cards keep missing, run "
                    "`python -m policysim.calibrate_camera` locally to retune detection "
                    "thresholds to your lighting (see README §8)."
                )

        card_photo = st.camera_input("Point at a card, then capture", label_visibility="collapsed", key="color_card_input")
        if card_photo is not None:
            # st.camera_input keeps returning the same object across reruns until the
            # user retakes a photo -- only act on a genuinely NEW capture, not every
            # unrelated rerun (e.g. clicking "Step round" elsewhere in the sidebar).
            if st.session_state.get("last_camera_photo_id") != card_photo.file_id:
                st.session_state["last_camera_photo_id"] = card_photo.file_id
                _handle_trigger_photo(card_photo.getvalue())

        with st.expander("Alternative: upload a photo instead of using the live camera"):
            uploaded = st.file_uploader("Upload a photo of a QR or color trigger card", type=["png", "jpg", "jpeg"])
            if uploaded is not None:
                if st.session_state.get("last_uploaded_photo_id") != uploaded.file_id:
                    st.session_state["last_uploaded_photo_id"] = uploaded.file_id
                    _handle_trigger_photo(uploaded.getvalue())

    st.markdown("---")
    gov = sim.government
    st.markdown("### Government state")
    st.caption(f"Tax rate: {gov.tax_rate*100:.0f}%")
    st.caption(f"Minimum wage: ₹{gov.minimum_wage:.2f}")
    st.caption(f"Firm subsidy rate: {gov.firm_subsidy_rate*100:.1f}%")
    st.caption(f"Cash transfer (low/mid): ₹{gov.cash_transfer_low:.0f} / ₹{gov.cash_transfer_mid:.0f}")

# ---------------------------------------------------------------- village HTML ---

def build_village_html(state: dict) -> str:
    # json.dumps does NOT escape "</script>" sequences, and this state
    # includes free-text reasoning strings (household/firm "last_reasoning")
    # that come from an LLM when the Groq backend is active -- i.e. this is
    # untrusted-ish generated text, not a static string literal we wrote.
    # If a model ever emitted "</script>" (accidentally or via a crafted
    # prompt reflected back into reasoning), it would break out of this
    # <script> tag and inject arbitrary HTML/JS into the page. Escaping the
    # forward slash (a JSON-safe, standard mitigation) neutralizes that
    # without changing the parsed value on the JS side.
    state_json = json.dumps(state).replace("</", "<\\/")
    return f"""
<!DOCTYPE html>
<html><head><meta charset="UTF-8" />
<style>
  @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@600;700;800&family=Manrope:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ width: 100%; height: 100%; background: #0f221e; font-family: 'Manrope', sans-serif; overflow: hidden; }}
  #scene {{ position: absolute; inset: 0; }}
  #header {{ position: absolute; top: 18px; left: 0; right: 0; text-align: center; color: #f5efe6; pointer-events: none; }}
  #header .eyebrow {{ font-size: 11px; letter-spacing: 0.18em; color: #9fb8b0; font-family: 'JetBrains Mono', monospace; }}
  #header h1 {{ font-family: 'Baloo 2', sans-serif; font-weight: 800; font-size: 26px; margin-top: 6px; }}
  #header h1 span {{ color: #e8935b; }}
  #drawer {{ position: absolute; top: 0; right: 0; height: 100%; width: min(310px, 92vw); background: rgba(15,34,30,0.94);
             backdrop-filter: blur(6px); border-left: 1px solid rgba(255,255,255,0.08); padding: 24px 22px;
             color: #f5efe6; overflow-y: auto; transform: translateX(100%); transition: transform .25s ease; }}
  #drawer.open {{ transform: translateX(0); }}
  #drawer .close {{ position: absolute; top: 14px; right: 16px; background: none; border: none; color: #9fb8b0; font-size: 20px; cursor: pointer; }}
  .eyebrow {{ font-size: 11px; letter-spacing: 0.15em; color: #9fb8b0; text-transform: uppercase; }}
  .title {{ font-family: 'Baloo 2', sans-serif; font-size: 22px; margin: 8px 0 16px; }}
  .stat-row {{ display: flex; justify-content: space-between; padding: 9px 0; border-bottom: 1px solid rgba(255,255,255,0.08); font-size: 13px; }}
  .stat-row span:first-child {{ color: #9fb8b0; }}
  .stat-row span:last-child {{ font-family: 'JetBrains Mono', monospace; font-weight: 600; }}
  .note {{ margin-top: 14px; font-size: 11px; color: #9fb8b0; font-family: 'JetBrains Mono', monospace; line-height: 1.5;
           border-top: 1px dashed rgba(255,255,255,0.12); padding-top: 12px; }}
  .policy-chip {{ display: block; width: 100%; margin-bottom: 8px; border-radius: 10px; padding: 9px 11px;
                  border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.05); color: #f5efe6; }}
  .policy-chip .label {{ font-family: 'Baloo 2', sans-serif; font-weight: 700; font-size: 12.5px; }}
  .policy-chip .desc {{ font-size: 10.5px; opacity: 0.85; margin-top: 2px; }}
  #footer {{ position: absolute; bottom: 12px; left: 0; right: 0; text-align: center; font-size: 10.5px; color: #9fb8b0;
             font-family: 'JetBrains Mono', monospace; letter-spacing: 0.05em; pointer-events: none; }}

  @media (max-width: 480px) {{
    #header h1 {{ font-size: 20px; }}
    #header .eyebrow {{ font-size: 9px; }}
    #footer {{ font-size: 9px; padding: 0 8px; }}
  }}
</style></head>
<body>
<div id="scene"></div>
<div id="header">
  <div class="eyebrow">POLICYSIM · ROUND {state['round']}</div>
  <h1>Econ Village <span>3D</span></h1>
</div>
<div id="drawer">
  <button class="close" id="drawer-close">✕</button>
  <div id="drawer-content"></div>
</div>
<div id="footer">drag to orbit · scroll to zoom · click a building · use the sidebar to step rounds & toggle policies</div>

<script type="importmap">
{{ "imports": {{ "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
"three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/" }} }}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const STATE = {state_json};
const TIER_WALL = {{ low: 0xf3d9b6, mid: 0xfbefe1, high: 0xece4d6 }};
const TIER_ROOF = {{ low: 0xe8935b, mid: 0xf2b880, high: 0xd9a441 }};
const POLICY_COLOR = {{
  subsidy_cut: '#e8604c', minimum_wage_increase: '#c9a227',
  luxury_tax: '#4c72a8', cash_transfer: '#6fae7a',
}};

let scene, camera, renderer, controls, raycaster, mouse;
let clickable = [];
let selected = null;

function initScene() {{
  const mount = document.getElementById('scene');
  const width = mount.clientWidth, height = mount.clientHeight;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f221e);
  scene.fog = new THREE.FogExp2(0x16302b, 0.011);

  camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 1000);
  camera.position.set(22, 27, 36);

  renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  mount.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 1, 4);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 14;
  controls.maxDistance = 60;
  controls.maxPolarAngle = Math.PI * 0.48;
  controls.update();

  scene.add(new THREE.AmbientLight(0x8fb8b0, 0.6));

  const key = new THREE.DirectionalLight(0xffe3b0, 1.15);
  key.position.set(24, 32, 16);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.left = -30; key.shadow.camera.right = 30;
  key.shadow.camera.top = 30; key.shadow.camera.bottom = -30;
  scene.add(key);

  const rim = new THREE.PointLight(0xe86a4c, 0.5, 60);
  rim.position.set(-14, 8, -14);
  scene.add(rim);

  const ground = new THREE.Mesh(
    new THREE.CircleGeometry(60, 64),
    new THREE.MeshStandardMaterial({{ color: 0x1d3a34, roughness: 1 }})
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);

  const plaza = new THREE.Mesh(
    new THREE.RingGeometry(9.5, 10.3, 48),
    new THREE.MeshStandardMaterial({{ color: 0xc9a227, roughness: 0.7 }})
  );
  plaza.rotation.x = -Math.PI / 2;
  plaza.position.set(0, 0.01, -6);
  scene.add(plaza);

  makeGov(0, -12);
  const houseCols = Math.ceil(Math.sqrt(STATE.households.length));
  STATE.households.forEach((hh, i) => {{
    const r = Math.floor(i / houseCols), c = i % houseCols;
    makeHouse((c - (houseCols - 1) / 2) * 4.1, 4 + r * 4.4, hh);
  }});
  const firmSpacing = STATE.firms.length > 2 ? 6.5 : 15;
  STATE.firms.forEach((f, i) => makeFirm((i - (STATE.firms.length - 1) / 2) * firmSpacing, -3.5, f, i === 0));

  raycaster = new THREE.Raycaster();
  mouse = new THREE.Vector2();

  let downX = 0, downY = 0;
  renderer.domElement.addEventListener('pointerdown', (e) => {{ downX = e.clientX; downY = e.clientY; }});
  renderer.domElement.addEventListener('pointerup', (e) => {{
    if (Math.abs(e.clientX - downX) > 4 || Math.abs(e.clientY - downY) > 4) return;
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(clickable);
    if (hits.length > 0) openDrawer(hits[0].object.userData);
  }});

  window.addEventListener('resize', () => resizeToMount());
  // window 'resize' alone misses a real case here: toggling Streamlit's
  // sidebar changes this component's width without the browser window
  // itself resizing, so the 3D view would silently keep rendering at the
  // stale aspect ratio. ResizeObserver watches the actual mount element.
  new ResizeObserver(() => resizeToMount()).observe(mount);

  function resizeToMount() {{
    const w = mount.clientWidth, h = mount.clientHeight;
    if (w === 0 || h === 0) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }}

  animate();
}}

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}

function tag(mesh, data) {{ mesh.userData = data; clickable.push(mesh); }}

function makeHouse(x, z, hh) {{
  const group = new THREE.Group();
  const wallMat = new THREE.MeshStandardMaterial({{ color: TIER_WALL[hh.tier] || 0xfbefe1, roughness: 0.85 }});
  const wall = new THREE.Mesh(new THREE.BoxGeometry(2.1, 1.6, 2.1), wallMat);
  wall.position.y = 0.8; wall.castShadow = true; wall.receiveShadow = true;
  group.add(wall);

  const roof = new THREE.Mesh(
    new THREE.ConeGeometry(1.65, 1.05, 4),
    new THREE.MeshStandardMaterial({{ color: TIER_ROOF[hh.tier] || 0xf2b880, roughness: 0.55 }})
  );
  roof.rotation.y = Math.PI / 4; roof.position.y = 1.6 + 0.5; roof.castShadow = true;
  group.add(roof);

  const winMat = new THREE.MeshStandardMaterial({{ color: 0xffe9b0, emissive: 0xffcf7a, emissiveIntensity: hh.employed ? 0.85 : 0.15 }});
  const win = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.4, 0.05), winMat);
  win.position.set(0, 0.9, 1.08);
  group.add(win);

  group.position.set(x, 0, z);
  const data = {{ type: 'household', id: hh.id }};
  tag(wall, data); tag(roof, data); tag(win, data);
  scene.add(group);
}}

function makeFirm(x, z, firm, isFirst) {{
  const group = new THREE.Group();
  const bodyMat = new THREE.MeshStandardMaterial({{ color: isFirst ? 0x4c848a : 0x386065, roughness: 0.6 }});
  const body = new THREE.Mesh(new THREE.BoxGeometry(3.8, 2.8, 3), bodyMat);
  body.position.y = 1.4; body.castShadow = true; body.receiveShadow = true;
  group.add(body);

  const capacity = Math.max(firm.employees, 1) * (firm.capacity_per_employee || 20);
  const utilization = capacity ? firm.last_demand / capacity : 0.5;
  const heat = Math.min(Math.max((utilization - 0.6) / 0.8, 0), 1);
  const chimneyMat = new THREE.MeshStandardMaterial({{ color: 0x2c4c50, emissive: 0xe8935b, emissiveIntensity: heat * 0.6 }});
  const chimney = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.34, 1.9, 12), chimneyMat);
  chimney.position.set(1.2, 3.7, 0.7); chimney.castShadow = true;
  group.add(chimney);

  group.position.set(x, 0, z);
  const data = {{ type: 'firm', id: firm.id }};
  tag(body, data); tag(chimney, data);
  scene.add(group);
}}

function makeGov(x, z) {{
  const group = new THREE.Group();
  const base = new THREE.Mesh(
    new THREE.BoxGeometry(6, 2.2, 3.8),
    new THREE.MeshStandardMaterial({{ color: 0xc9a227, roughness: 0.45, metalness: 0.2 }})
  );
  base.position.y = 1.1; base.castShadow = true; base.receiveShadow = true;
  group.add(base);
  const data = {{ type: 'gov', id: 'gov' }};
  tag(base, data);

  for (let i = -2; i <= 2; i += 1.6) {{
    if (Math.abs(i) < 0.3) continue;
    const col = new THREE.Mesh(
      new THREE.CylinderGeometry(0.17, 0.17, 2.1, 12),
      new THREE.MeshStandardMaterial({{ color: 0xf5efe6 }})
    );
    col.position.set(i, 1.05, 1.95); col.castShadow = true;
    tag(col, data);
    group.add(col);
  }}

  const roof = new THREE.Mesh(
    new THREE.ConeGeometry(4.4, 1.5, 4),
    new THREE.MeshStandardMaterial({{ color: 0x9c7e1d, roughness: 0.5 }})
  );
  roof.rotation.y = Math.PI / 4; roof.position.y = 2.2 + 0.75; roof.castShadow = true;
  tag(roof, data);
  group.add(roof);

  group.position.set(x, 0, z);
  scene.add(group);
}}

function openDrawer(data) {{
  selected = data;
  document.getElementById('drawer').classList.add('open');
  renderDrawer();
}}
document.getElementById('drawer-close').addEventListener('click', () => {{
  selected = null;
  document.getElementById('drawer').classList.remove('open');
}});

function statRow(label, value) {{
  return `<div class="stat-row"><span>${{label}}</span><span>${{value}}</span></div>`;
}}

// last_reasoning is LLM-generated free text when the Groq backend is
// active -- treat it like any other untrusted string before it goes into
// innerHTML, so a model output containing HTML/script can't execute.
function escapeHtml(str) {{
  const div = document.createElement('div');
  div.textContent = str == null ? '' : String(str);
  return div.innerHTML;
}}

function renderDrawer() {{
  const el = document.getElementById('drawer-content');
  if (!selected) return;

  if (selected.type === 'household') {{
    const hh = STATE.households.find((h) => h.id === selected.id);
    el.innerHTML = `
      <div class="eyebrow">Household · ${{hh.tier}} income</div>
      <div class="title">H-${{String(hh.id).padStart(2, '0')}}</div>
      ${{statRow('Employed', hh.employed ? 'Yes' : 'No')}}
      ${{statRow('Disposable income', `₹${{(hh.last_income || hh.base_income).toFixed(0)}}`)}}
      ${{statRow('Last round spend', `₹${{hh.last_spend.toFixed(0)}}`)}}
      ${{statRow('Savings', `₹${{hh.savings.toFixed(0)}}`)}}
      <div class="note">${{escapeHtml(hh.last_reasoning) || "Hasn't reasoned yet — step a round from the sidebar."}}</div>
    `;
  }} else if (selected.type === 'firm') {{
    const firm = STATE.firms.find((f) => f.id === selected.id);
    el.innerHTML = `
      <div class="eyebrow">${{firm.kind === 'small_shop' ? 'Firm · small shop' : 'Firm · large firm'}}</div>
      <div class="title">${{firm.name}}</div>
      ${{statRow('Price', `₹${{firm.price.toFixed(2)}}`)}}
      ${{statRow('Wage', `₹${{firm.wage.toFixed(2)}}/round`)}}
      ${{statRow('Employees', `${{firm.employees}} / ${{firm.max_employees}}`)}}
      ${{statRow('Last demand', firm.last_demand.toFixed(1))}}
      ${{statRow('Last revenue', `₹${{firm.last_revenue.toFixed(0)}}`)}}
      <div class="note">${{escapeHtml(firm.last_reasoning) || "Hasn't reasoned yet — step a round from the sidebar."}}</div>
    `;
  }} else {{
    const gov = STATE.government;
    const chips = Object.entries(STATE.policy_library).map(([id, spec]) => {{
      const active = !!gov.active_policies[id];
      const color = POLICY_COLOR[id] || '#9fb8b0';
      return `<div class="policy-chip" style="background:${{active ? color : 'rgba(255,255,255,0.05)'}}; border-color:${{active ? color : 'rgba(255,255,255,0.12)'}}; color:${{active ? '#16241F' : '#f5efe6'}};">
                <div class="label">${{active ? '🟢' : '⚪'}} ${{spec.label}}</div>
                <div class="desc">${{spec.description}}</div>
              </div>`;
    }}).join('');
    el.innerHTML = `
      <div class="eyebrow">Policymaker</div>
      <div class="title">Town Hall</div>
      ${{statRow('Tax rate', `${{(gov.tax_rate * 100).toFixed(0)}}%`)}}
      ${{statRow('Minimum wage', `₹${{gov.minimum_wage.toFixed(2)}}`)}}
      ${{statRow('Firm subsidy rate', `${{(gov.firm_subsidy_rate * 100).toFixed(1)}}%`)}}
      ${{statRow('Cash transfer (low/mid)', `₹${{gov.cash_transfer_low.toFixed(0)}} / ₹${{gov.cash_transfer_mid.toFixed(0)}}`)}}
      <div style="margin-top:16px; font-size:11px; color:#9fb8b0; margin-bottom:8px;">Toggle these from the sidebar — this panel is read-only.</div>
      ${{chips}}
    `;
  }}
}}

initScene();
</script>
</body></html>
"""


village_state = snapshot(sim)

if sim.history:
    import pandas as pd

    hist_df = pd.DataFrame(sim.history).set_index("round")
    chart_col1, chart_col2, chart_col3 = st.columns(3)
    with chart_col1:
        st.caption("Price index")
        st.line_chart(hist_df["price_level"], height=140)
    with chart_col2:
        st.caption("Unemployment rate")
        st.line_chart(hist_df["unemployment_rate"], height=140)
    with chart_col3:
        st.caption("Wealth Gini")
        st.line_chart(hist_df["gini"], height=140)
else:
    st.caption("Step a round from the sidebar to start seeing price, unemployment, and inequality trends here.")

components.html(build_village_html(village_state), height=680, scrolling=False)
