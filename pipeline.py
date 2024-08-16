#!/usr/bin/env python

# IMPORTS

import os
import sys
import time
import shutil
import logging
import argparse
import numpy as np
import nibabel as nib
from types import SimpleNamespace

import ants
import torch

import ddsp.seg
import ddsp.surface
import ddsp.sphere

from ddsp import template
from ddsp.utils.timer import Timer
from ddsp.volume.volume import Volume

from ddsp.sphere.net.loss import (
    edge_distortion,
    area_distortion,
)

from ddsp.utils.inflate import (
    generate_inflated_surfaces,
    wb_generate_inflated_surfaces,
)
from ddsp.utils.io import (
    save_numpy_to_nifti,
    save_gifti_surface,
    save_gifti_metric,
    create_wb_spec,
)
from ddsp.utils.mesh import (
    apply_affine_mat,
    cot_laplacian_smooth,
    laplacian_smooth,
)
from ddsp.utils.metric import (
    metric_dilation,
    cortical_thickness,
    curvature,
    sulcal_depth,
    myelin_map,
    smooth_myelin_map,
)
from ddsp.utils.register import (
    registration,
)


# DEFINITIONS

LOGGER_NAME = "dhcp-dl-surface_pipeline"
LOGLEVEL = 1
LOGMODE = "w"
EPSILON = 1e-12
MAX_REGIST_ITER = 5
MIN_REGIST_DICE = 0.9


# IMPLEMENTATION

def main(args):

    if not verify_inputs(args):
        raise AssertionError("Missing inputs.")

    # Run pipeline
    t_start = time.time()
    logger.info("Starting pipeline...")

    # Load data
    with Timer(print_fn=logger.info) as timer:
        logger.info("Loading inputs...")
        t1_original, t2_original = load_data(args)
        logger.info("Finished loading inputs.")

    # Load templates
    templates = SimpleNamespace(
        t2_atlas=template.load_t2_atlas(),
        initial_surfaces=template.load_surface_template(),
        input_spheres=template.load_input_sphere(args.device),
        template_sphere=template.load_template_sphere(args.device),
        barycentric_coordinates=template.load_barycentric_coordinates(),
    )

    # Compute T1w/T2w ratio
    if args.t1 is not None:
        with Timer(print_fn=logger.info) as timer:
            logger.info("Computing T1w/T2w ratio...")
            t1t2_ratio = compute_t1t2_ratio(t1_original, t2_original)
            logger.info("Finished computing T1w/T2w ratio.")
            # Save T1w/T2w ratio as a NIfTI volume
            try:
                ratio_output = os.path.join(
                    args.out_dir, "T1wDividedByT2w.nii.gz"
                )
                save_numpy_to_nifti(
                    t1t2_ratio, t2_original.affine(), ratio_output
                )
            except Exception as exc:
                # Non-breaking exception
                logger.error(
                    f"ERROR: Failed to save T1w/T2w ratio: {ratio_output}"
                )
                logger.error(str(exc))
            else:
                logger.info(f"SAVED: {ratio_output}")

    # Bias field correction
    with Timer(print_fn=logger.info) as timer:
        logger.info("Starting bias field correction...")
        t2_restore = bias_field_correction(t2_original, args.verbose)
        logger.info("Completed T2w bias field correction.")
        # Save brain-extracted and bias-corrected T2w image
        try:
            restore_output = os.path.join(
                args.out_dir, "T2w_restore_brain.nii.gz"
            )
            save_numpy_to_nifti(
                t2_restore.data(), t2_original.affine(), restore_output
            )
        except Exception as exc:
            # Non-breaking exception
            logger.error(
                f"ERROR: Failed to save bias-corrected T2w: {restore_output}"
            )
            logger.error(str(exc))
        else:
            logger.info(f"SAVED: {restore_output}")

    # Affine registration
    with Timer(print_fn=logger.info) as timer:
        logger.info("Starting affine registration...")
        t2_aligned, antstx = affine_registration(
            t2_restore,
            templates.t2_atlas,
            outputdir=args.out_dir,
            verbose=args.verbose,
        )
        logger.info("Completed affine registration.")

    # Cortical ribbon segmentation
    with Timer(print_fn=logger.info) as timer:
        logger.info("Cortical ribbon segmentation starts...")
        ribbon_original = cortical_ribbon_segmentation(
            t2_original=t2_original,
            t2_aligned=t2_aligned,
            t2_atlas=templates.t2_atlas,
            antstx=antstx,
            device=args.device,
        )
        logger.info("Completed cortical ribbon segmentation.")
        # Save ribbon segmentation to a NIfTI volume
        try:
            ribbon_output = os.path.join(args.out_dir, "ribbon.nii.gz")
            save_numpy_to_nifti(
                ribbon_original, t2_original.affine(), ribbon_output
            )
        except Exception as exc:
            # Non-breaking exception
            logger.error(
                f"ERROR: Failed to save cortical ribbon segmentation: "
                f"{ribbon_original}"
            )
            logger.error(str(exc))
        else:
            logger.info(f"SAVED: {ribbon_output}")

    # Cortical surface reconstruction
    with Timer(print_fn=logger.info) as timer:
        for hemi in ("left", "right"):
            logger.info(f"Starting ({hemi}) cortical surface reconstruction...")
            vertices, vert_wm, vert_wm_orig, faces = surface_reconstruction(
                t2_aligned,
                templates,
                hemi=hemi,
                outputdir=args.out_dir,
                device=args.device,
            )
        else:
            logger.info(f"Completed cortical surface reconstruction.")

    # Surface inflation
    with Timer(print_fn=logger.info) as timer:
        for hemi in ("left", "right"):
            logger.info(f"Starting ({hemi}) cortical surface inflation...")
            inflated_vertices, inflated_faces = surface_inflation(
                vertices,
                faces,
                hemi=hemi,
                outputdir=args.out_dir,
                device=args.device,
            )
        else:
            logger.info(f"Completed cortical surface inflation.")

    # Spherical projection
    with Timer(print_fn=logger.info) as timer:
        for hemi in ("left", "right"):
            logger.info(f"Starting ({hemi}) spherical projection...")
            spherical_projection(
                vertices,
                vert_wm_orig,
                faces,
                templates,
                hemi=hemi,
                outputdir=args.out_dir,
                device=args.device,
            )
        else:
            logger.info(f"Completed spherical projection.")

    # Cortical feature estimation
    with Timer(print_fn=logger.info) as timer:
        for hemi in ("left", "right"):
            logger.info(f"Starting ({hemi}) cortical feature estimation...")
            cortical_feature_estimation(
                t1_original,
                vertices,
                faces,
                hemi=hemi,
                outputdir=args.out_dir,
                device=args.device,
            )
        else:
            logger.info(f"Completed cortical feature estimation.")

    # Conclusion
    t_end = time.time()
    conclude(outputdir=args.out_dir)
    logger.info("----------------------------------------")
    logger.info(
        "Finished. Total runtime: {} sec.".format(np.round(t_end - t_start, 4))
    )
    logger.info("========================================")


def verify_inputs(args) -> bool:
    """
    Ensure that at least one of the inputs exists.

    """
    inputs_ok = False

    # T1
    if args.t1 is None or not os.path.exists(args.t1):
        logger.warning(f"T1 input does not exist: {args.t2}")
    else:
        args.t1 = os.path.abspath(args.t1)
        inputs_ok = True

    # T2
    if args.t2 is None or not os.path.exists(args.t2):
        logger.warning(f"T2 input does not exist: {args.t2}")
    else:
        args.t2 = os.path.abspath(args.t2)
        inputs_ok = True

    return inputs_ok


def load_data(args) -> tuple[Volume, Volume]:

    # T2
    logger.info("Loading T2 image ...")
    t2_orig = Volume(
        antsvol=ants.image_read(args.t2),
        niftiheader=nib.load(args.t2).header,
    )
    logger.info("Done.")

    # Load brain mask if it exists
    if args.mask is not None:
        logger.info("Loading brain mask...")
        t2_orig.mask = nib.load(args.mask).get_fdata()
        logger.info("Done.")

    # T1
    if args.t1 is not None:
        logger.info("Loading T1 image ...")
        t1_orig = Volume(
            antsvol=ants.image_read(args.t1),
            niftiheader=nib.load(args.t1).header,
            mask=t2_orig.mask,
        )
        logger.info("Done.")
    else:
        t1_orig = None

    return t1_orig, t2_orig


def compute_t1t2_ratio(t1: Volume, t2: Volume) -> np.ndarray:
    ratio = np.clip(t1.data() / (t2.data() + EPSILON), 0, 100)
    if t2.mask is not None:
        ratio = ratio * t2.mask
    return ratio


def bias_field_correction(t2_original: Volume, verbose: bool = False) -> Volume:
    """
    N4 bias field correction.

    """
    t2_restore_ants = ants.utils.bias_correction.n4_bias_field_correction(
        t2_original.antsvol(apply_mask=False),
        mask=t2_original.antsvol(data=t2_original.mask, apply_mask=False),
        shrink_factor=4,
        convergence={"iters": [50, 50, 50], "tol": 0.001},
        spline_param=100,
        verbose=verbose,
    )
    return Volume(
        antsvol=t2_restore_ants,
        niftiheader=t2_original.header,
        mask=t2_original.mask,
    )


def affine_registration(
    t2_restore: Volume, t2_atlas: Volume, outputdir: str, verbose: bool = False
) -> tuple[Volume, SimpleNamespace]:

    # ANTs affine registration
    img_t2_align_ants, affine_t2_align, ants_rigid, ants_affine, align_dice = (
        registration(
            img_move_ants=t2_restore.antsvol(apply_mask=True),
            img_fix_ants=t2_atlas.antsvol(apply_mask=False),
            affine_fix=t2_atlas.affine(),
            out_prefix=outputdir,
            max_iter=MAX_REGIST_ITER,
            min_dice=MIN_REGIST_DICE,
            verbose=verbose,
        )
    )

    # Check Dice score
    if align_dice >= MIN_REGIST_DICE:
        logger.info(
            f"SUCCESS: Dice after registration: {align_dice} "
            f">= {MIN_REGIST_DICE}"
        )
    else:
        logger.error(
            f"ERROR: Expected Dice>{MIN_REGIST_DICE} after affine "
            f"registration, got Dice={align_dice:.03f}."
        )

    # Create NIfTI header using the affine
    hdr_aligned = nib.Nifti1Header()
    hdr_aligned.set_sform(affine_t2_align, code=2)

    # Generate output
    t2_aligned = Volume(
        antsvol=img_t2_align_ants,
        niftiheader=hdr_aligned,  # TODO: verify this
        mask=t2_restore.mask,
    )
    ants_transformations = SimpleNamespace(rigid=ants_rigid, affine=ants_affine)

    return t2_aligned, ants_transformations


def cortical_ribbon_segmentation(
    t2_original: Volume,
    t2_aligned: Volume,
    t2_atlas: Volume,
    antstx: SimpleNamespace,
    device: str,
) -> Volume:

    # Input volume for nn model
    tensor_t2_aligned = torch.Tensor(t2_aligned.data()[None, None]).to(device)
    tensor_t2_aligned = (tensor_t2_aligned / tensor_t2_aligned.max()).float()
    tensor_in = tensor_t2_aligned.clone()

    # Load cortical ribbon segmentation model
    seg_model = ddsp.seg.load(device)

    # Predict cortical ribbon
    with torch.no_grad():
        ribbon_pred = torch.sigmoid(seg_model(tensor_in))
    ribbon_aligned_ants = t2_aligned.antsvol(
        data=ribbon_pred[0, 0].cpu().numpy(), apply_mask=False
    )

    # Transform back to original space
    ribbon_orig_ants = ants.apply_transforms(
        fixed=t2_atlas.antsvol(apply_mask=False),
        moving=ribbon_aligned_ants,
        transformlist=antstx.affine["invtransforms"],
        whichtoinvert=[True],
        interpolator="linear",
    )
    ribbon_orig_ants = ants.apply_transforms(
        fixed=t2_original.antsvol(apply_mask=False),
        moving=ribbon_orig_ants,
        transformlist=antstx.rigid["invtransforms"],
        whichtoinvert=[True],
        interpolator="linear",
    )

    # Threshold to create a binary ribbon segmentation
    ribbon_original = t2_original.antsvol(
        data=(ribbon_orig_ants.numpy() > 0.5).astype(np.float32),
        apply_mask=False,
    )

    return Volume(
        antsvol=ribbon_original,
        niftiheader=t2_original.header,
        mask=t2_original.mask,
    )


def surface_reconstruction(
    t2_aligned: Volume,
    templates: SimpleNamespace,
    hemi: str,
    outputdir: str,
    device: str,
):

    # Set model
    surf_recon_models = ddsp.surface.load(device=device)
    surf_recon_wm = vars(surf_recon_models)[hemi].wm
    surf_recon_pial = vars(surf_recon_models)[hemi].pial

    # Volume
    vol_in = torch.Tensor(t2_aligned.data()[None, None]).to(device)
    vol_in = dict(left=vol_in[:, :, 64:], right=vol_in[:, :, :112])[hemi]

    # Vertices
    surf_in = vars(templates.initial_surfaces)[hemi]
    vert_in = surf_in.agg_data("pointset")
    vert_in = apply_affine_mat(
        vert_in, np.linalg.inv(templates.t2_atlas.affine())
    )
    vert_in = dict(left=vert_in - [64, 0, 0], right=vert_in)[hemi]
    vert_in = torch.Tensor(vert_in[None]).to(device)

    # Faces
    face_in = surf_in.agg_data("triangle")
    face_in = face_in[:, [2, 1, 0]]
    face_in = torch.LongTensor(face_in[None]).to(device)

    # WM and pial surface reconstruction
    with torch.no_grad():
        vert_wm = surf_recon_wm(vert_in, vol_in, n_steps=7)
        vert_wm = cot_laplacian_smooth(vert_wm, face_in, n_iters=1)
        vert_pial = surf_recon_pial(vert_wm, vol_in, n_steps=7)
        vert_pial = laplacian_smooth(vert_pial, face_in, n_iters=1)

    # torch.Tensor -> numpy.array
    vert_wm_align = vert_wm[0].cpu().numpy()
    vert_pial_align = vert_pial[0].cpu().numpy()
    face_align = face_in[0].cpu().numpy()

    # Transform vertices to original space
    vert_wm_orig = dict(
        left=vert_wm_align + [64, 0, 0],
        right=vert_wm_align.copy(),
    )[hemi]
    vert_pial_orig = dict(
        left=vert_pial_align + [64, 0, 0],
        right=vert_pial_align.copy(),
    )[hemi]
    vert_wm_orig = apply_affine_mat(vert_wm_orig, t2_aligned.affine())
    vert_pial_orig = apply_affine_mat(vert_pial_orig, t2_aligned.affine())
    face_orig = face_align[:, [2, 1, 0]]

    # Mid-thickness surface
    vert_mid_orig = (vert_wm_orig + vert_pial_orig) / 2

    # Save surfaces in GIfTI format
    # WM surface
    save_gifti_surface(
        vert_wm_orig,
        face_orig,
        gifti_path=os.path.join(outputdir, f"hemi-{hemi}_wm.surf.gii"),
        hemi=hemi,
        surf_type="wm",
    )
    # Pial surface
    save_gifti_surface(
        vert_pial_orig,
        face_orig,
        gifti_path=os.path.join(outputdir, f"hemi-{hemi}_pial.surf.gii"),
        hemi=hemi,
        surf_type="pial",
    )
    # Mid-thickness surface
    save_gifti_surface(
        vert_mid_orig,
        face_orig,
        gifti_path=os.path.join(
            outputdir, f"hemi-{hemi}_midthickness.surf.gii"
        ),
        hemi=hemi,
        surf_type="midthickness",
    )

    # Generate output: on-device tensors for further processing
    vertices = SimpleNamespace(
        wm=torch.Tensor(vert_wm_orig).unsqueeze(0).to(device),
        pial=torch.Tensor(vert_pial_orig).unsqueeze(0).to(device),
        mid=torch.Tensor(vert_mid_orig).unsqueeze(0).to(device),
    )
    faces = torch.LongTensor(face_orig).unsqueeze(0).to(device)

    return vertices, vert_wm, vert_wm_orig, faces


def surface_inflation(
    vertices: SimpleNamespace,
    faces: torch.LongTensor,
    hemi: str,
    outputdir: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create inflated and very inflated surfaces.

    """
    # If device is cpu, use wb_command for inflation (faster)
    if device == "cpu":
        vert_inflated_orig, vert_vinflated_orig = wb_generate_inflated_surfaces(
            outputdir, hemi, iter_scale=3.0
        )
    else:  # cuda acceleration
        vert_inflated, vert_vinflated = generate_inflated_surfaces(
            vertices.mid, faces, iter_scale=3.0
        )
        vert_inflated_orig = vert_inflated[0].cpu().numpy()
        vert_vinflated_orig = vert_vinflated[0].cpu().numpy()

    # Save inflated surfaces as .surf.gii
    face_orig = faces[0].cpu().numpy()
    save_gifti_surface(
        vert_inflated_orig,
        face_orig,
        gifti_path=os.path.join(outputdir, f"hemi-{hemi}_inflated.surf.gii"),
        hemi=hemi,
        surf_type="inflated",
    )
    save_gifti_surface(
        vert_vinflated_orig,
        face_orig,
        gifti_path=os.path.join(outputdir, f"hemi-{hemi}_vinflated.surf.gii"),
        hemi=hemi,
        surf_type="vinflated",
    )

    return vert_inflated_orig, vert_vinflated_orig


def spherical_projection(
    vertices: SimpleNamespace,
    vert_wm_orig,
    faces: torch.LongTensor,
    templates: SimpleNamespace,
    hemi: str,
    outputdir: str,
    device: str,
):
    # Set model, input vertices and faces
    sp_model = ddsp.sphere.load(device)
    vert_sphere_in = getattr(templates.input_spheres, hemi)
    bc_coord = getattr(templates.barycentric_coordinates.vertices, hemi)
    face_id = getattr(templates.barycentric_coordinates.faces, hemi)
    vert_sphere_160k = templates.template_sphere.vertices

    # Interpolate to 160k template
    vert_wm_160k = (vert_wm_orig[face_id] * bc_coord[..., None]).sum(-2)
    vert_wm_160k = torch.Tensor(vert_wm_160k[None]).to(device)
    feat_160k = torch.cat([vert_sphere_160k, vert_wm_160k], dim=-1)

    with torch.no_grad():
        vert_sphere = sp_model(feat_160k, vert_sphere_in, n_steps=7)

    # compute metric distortion
    edge = torch.cat(
        [faces[0, :, [0, 1]], faces[0, :, [1, 2]], faces[0, :, [2, 0]]], dim=0
    ).T
    edge_distort = (
        100.0 * edge_distortion(vert_sphere, vertices.wm, edge).item()
    )
    area_distort = (
        100.0 * area_distortion(vert_sphere, vertices.wm, faces).item()
    )
    logger.info("Edge distortion: {}%".format(np.round(edge_distort, 2)))
    logger.info("Area distortion: {}%".format(np.round(area_distort, 2)))

    # save as .surf.gii
    vert_sphere = vert_sphere[0].cpu().numpy()
    save_gifti_surface(
        vert_sphere,
        faces.numpy()[0],
        gifti_path=os.path.join(outputdir, f"hemi-{hemi}_sphere.surf.gii"),
        hemi=hemi,
        surf_type="sphere",
    )


def cortical_feature_estimation(
    t1_original,
    vertices,
    faces: torch.LongTensor,
    hemi: str,
    outputdir: str,
    device: str,
):
    logger.info("Estimating cortical thickness...")
    thickness = cortical_thickness(vertices.wm, vertices.pial)
    thickness = metric_dilation(
        torch.Tensor(thickness[None, :, None]).to(device),
        faces,
        n_iters=10,
    )
    save_gifti_metric(
        metric=thickness,
        gifti_path=os.path.join(outputdir, f"hemi-{hemi}_thickness.shape.gii"),
        hemi=hemi,
        metric_type="thickness",
    )
    logger.info("Done.")

    logger.info("Estimating curvature...")
    curv = curvature(vertices.wm, faces, smooth_iters=5)
    save_gifti_metric(
        metric=curv,
        gifti_path=os.path.join(outputdir, f"hemi-{hemi}_curv.shape.gii"),
        hemi=hemi,
        metric_type="curv",
    )
    logger.info("Done.")

    logger.info("Estimating sulcal depth...")
    sulc = sulcal_depth(vertices.wm, faces, verbose=False)
    save_gifti_metric(
        metric=sulc,
        gifti_path=os.path.join(outputdir, f"hemi-{hemi}_sulc.shape.gii"),
        hemi=hemi,
        metric_type="sulc",
    )
    logger.info("Done.")

    # Estimate myelin map, based on the T1w/T2w ratio,
    # the mid-thickness surface, the cortical thickness,
    # and the cortical ribbon.
    if t1_original is not None:
        logger.info("Estimating myelin map...")
        myelin = myelin_map(subj_dir=outputdir, hemi=hemi)
        # metric dilation
        myelin = metric_dilation(
            torch.Tensor(myelin[None, :, None]).to(device),
            faces,
            n_iters=10,
        )
        # save myelin map
        save_gifti_metric(
            metric=myelin,
            gifti_path=os.path.join(
                outputdir, f"hemi-{hemi}_myelinmap.shape.gii"
            ),
            hemi=hemi,
            metric_type="myelinmap",
        )

        # smooth myelin map
        smoothed_myelin = smooth_myelin_map(subj_dir=outputdir, hemi=hemi)
        save_gifti_metric(
            metric=smoothed_myelin,
            gifti_path=os.path.join(
                outputdir, f"hemi-{hemi}_smoothed_myelinmap.shape.gii"
            ),
            hemi=hemi,
            metric_type="smoothed_myelinmap",
        )
        logger.info("Done.")


def conclude(outputdir: str):
    # Clean temporary files
    os.remove(os.path.join(outputdir, "rigid_0GenericAffine.mat"))
    os.remove(os.path.join(outputdir, "affine_0GenericAffine.mat"))
    os.remove(os.path.join(outputdir, "ribbon.nii.gz"))
    try:
        os.remove(os.path.join(outputdir, "T1wDividedByT2w.nii.gz"))
    except FileNotFoundError:
        pass

    # Create .spec file for visualization
    create_wb_spec(outputdir)


def create_cli():
    parser = argparse.ArgumentParser(description="dHCP DL Surface Pipeline")
    parser.add_argument(
        "--t2", default=None, type=str, help="Suffix of T2 image file."
    )
    parser.add_argument(
        "--t1", default=None, type=str, help="Suffix of T1 image file."
    )
    parser.add_argument(
        "--mask", default=None, type=str, help="Suffix of brain mask file."
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        type=str,
        help="Directory for saving the output of the pipeline.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Device for running the pipeline: [cuda, cpu]",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print debugging information."
    )
    return parser


def configure_output_directory(args):
    args.out_dir = os.path.abspath(args.out_dir)
    if os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    else:
        os.makedirs(args.out_dir, exist_ok=False)


def create_logger(
    outputdir: str,
    verbose: bool = False,
    level: int = LOGLEVEL,
    mode: str = LOGMODE,
):
    """
    Creates a suitably configured Logger instance.

    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers = []  # delete any existing handlers to avoid duplicate logs
    logger.setLevel(1)
    formatter = logging.Formatter(
        fmt="%(asctime)s Process-%(process)d %(levelname)s (%(lineno)d) "
        "- %(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
    )

    # Redirect all logs of interest to the specified logfile
    logfile = os.path.join(outputdir, "logfile.log")
    fh = logging.FileHandler(logfile, mode=mode, delay=False)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Redirect only warnings/errors to the standard output, unless "verbose"
    ch = logging.StreamHandler()
    if verbose:
        ch.setLevel(1)
    else:
        ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


# ------ Run dHCP DL-based surface pipeline ------
if __name__ == "__main__":
    parser = create_cli()
    if len(sys.argv) > 1:
        args = parser.parse_args()
        configure_output_directory(args)
        logger = create_logger(args.out_dir, args.verbose)
        main(args)
    else:
        parser.print_help()
