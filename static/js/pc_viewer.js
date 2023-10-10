import * as THREE from "three";
import { PCDLoader } from "three/addons/loaders/PCDLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GUI } from "three/addons/libs/lil-gui.module.min.js";

class ThreeJSPCDViewer extends HTMLElement {
    constructor() {
        super();
        const pcdPath = this.dataset.path;
        let camera, scene, renderer;
        let render_width = this.querySelector("div").clientWidth;
        let render_height = render_width;
        const parent_elem = this.querySelector("div");
        init();
        render();

        function init() {
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.setSize(render_width, render_height);
            renderer.set
            parent_elem.appendChild(renderer.domElement);
            // document.body.appendChild(renderer.domElement);

            scene = new THREE.Scene();

            camera = new THREE.PerspectiveCamera(
                30,
                render_width / render_height,
                0.01,
                40
            );
            camera.position.set(-15, 5, -2);
            scene.add(camera);

            const controls = new OrbitControls(camera, renderer.domElement);
            controls.addEventListener("change", render); // use if there is no animation loop
            controls.minDistance = 10.0;
            controls.maxDistance = 50.0;

            //scene.add( new THREE.AxesHelper( 1 ) );

            const loader = new PCDLoader();
            loader.load(pcdPath, function (points) {
                points.geometry.center();
                points.geometry.rotateX(Math.PI);
                points.material.size = 0.05;
                points.name = "figure.pcd";
                scene.add(points);
                render();
            });

            window.addEventListener("resize", onWindowResize);
        }

        function onWindowResize() {
            render_width = this.querySelector("div").clientWidth;
            render_height = this.querySelector("div").clientHeight;

            camera.aspect = render_width / render_height;
            camera.updateProjectionMatrix();

            renderer.setSize(render_width, render_height);

            render();
        }

        function render() {
            renderer.render(scene, camera);
        }
    }
}

customElements.define("three-js-pcd-viewer", ThreeJSPCDViewer);