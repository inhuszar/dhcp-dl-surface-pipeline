#!/usr/bin/env python

# IMPORTS

import argparse
import logging
import os
import shutil
import sys
import time
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch

import ants

import ddsp.seg
import ddsp.surface
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
    logger.info("Starting pipeline...")

    # Load data
    with Timer(print_fn=logger.info) as timer:
        logger.info("Loading inputs...")
        t1_original, t2_original = load_data(args)
        logger.info("Finished loading inputs.")

    # Load templates
    templates = SimpleNamespace(
        t2_atlas=template.load_t2_atlas(),
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
                    args.out_dir, "T1wDividedByT2w.nii.gz")
                save_numpy_to_nifti(
                    t1t2_ratio, t2_original.affine(), ratio_output)
            except Exception as exc:
                # Non-breaking exception
                logger.error(
                    f"ERROR: Failed to save T1w/T2w ratio: {ratio_output}")
                logger.error(str(exc))
            else:
                logger.info(f"SAVED: {ratio_output}")

    # Bias field correction
    with Timer(print_fn=logger.info) as timer:
        logger.info('Starting bias field correction...')
        t2_restore = bias_field_correction(t2_original, args.verbose)
        logger.info("Completed T2 bias field correction.")
        # Save brain-extracted and bias-corrected T2w image
        try:
            restore_output = os.path.join(
                args.out_dir, "T2w_restore_brain.nii.gz")
            save_numpy_to_nifti(
                t2_restore.data(), t2_original.affine(), restore_output)
        except Exception as exc:
            # Non-breaking exception
            logger.error(
                f"ERROR: Failed to save bias-corrected T2: {restore_output}")
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
                ribbon_original, t2_original.affine(), ribbon_output)
        except Exception as exc:
            # Non-breaking exception
            logger.error(
                f"ERROR: Failed to save cortical ribbon segmentation: "
                f"{ribbon_original}")
            logger.error(str(exc))
        else:
            logger.info(f"SAVED: {ribbon_output}")

    # Cortical surface reconstruction
    with Timer(print_fn=logger.info) as timer:
        for hemi in ("left", "right"):
            logger.info(f"Starting ({hemi}) cortical surface reconstruction...")
            surface_reconstruction(t2_aligned, hemi=hemi, device=args.device)
        else:
            logger.info(f"Completed cortical surface reconstruction.")


def verify_inputs(args) -> bool:
    """
    Ensure that at least one of the inputs exists.

    """
    inputs_ok = False

    # T1
    if args.t1 is None or not os.path.exists(args.t1):
        logger.warning(f"T1 input does not exist: {args.t2}")
        inputs_ok = True
    else:
        args.t1 = os.path.abspath(args.t1)

    # T2
    if args.t2 is None or not os.path.exists(args.t2):
        logger.warning(f"T2 input does not exist: {args.t2}")
        inputs_ok = True
    else:
        args.t2 = os.path.abspath(args.t2)

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
        convergence={
            "iters": [50, 50, 50],
            "tol": 0.001
        },
        spline_param=100,
        verbose=verbose
    )
    return Volume(
        antsvol=t2_restore_ants,
        niftiheader=t2_original.header,
        mask=t2_original.mask,
    )


def affine_registration(
        t2_restore: Volume,
        t2_atlas: Volume,
        outputdir: str,
        verbose: bool = False
) -> tuple[Volume, SimpleNamespace]:

    # ANTs affine registration
    img_t2_align_ants, affine_t2_align, ants_rigid, \
        ants_affine, align_dice = registration(
        img_move_ants=t2_restore.antsvol(apply_mask=True),
        img_fix_ants=t2_atlas.antsvol(apply_mask=False),
        affine_fix=t2_atlas.affine(),
        out_prefix=outputdir,
        max_iter=MAX_REGIST_ITER,
        min_dice=MIN_REGIST_DICE,
        verbose=verbose,
    )

    # Check Dice score
    if align_dice >= MIN_REGIST_DICE:
        logger.info(
            f"SUCCESS: Dice after registration: {align_dice} "
            f">= {MIN_REGIST_DICE}")
    else:
        logger.error(
            f"ERROR: Expected Dice>{MIN_REGIST_DICE} after affine "
            f"registration, got Dice={align_dice}.")

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
        device: str
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
        data=ribbon_pred[0, 0].cpu().numpy(), apply_mask=False)

    # Transform back to original space
    ribbon_orig_ants = ants.apply_transforms(
        fixed=t2_atlas.antsvol(apply_mask=False),
        moving=ribbon_aligned_ants,
        transformlist=antstx.affine["invtransforms"],
        whichtoinvert=[True],
        interpolator="linear"
    )
    ribbon_orig_ants = ants.apply_transforms(
        fixed=t2_original.antsvol(apply_mask=False),
        moving=ribbon_orig_ants,
        transformlist=antstx.rigid["invtransforms"],
        whichtoinvert=[True],
        interpolator="linear"
    )

    # Threshold to create a binary ribbon segmentation
    ribbon_original = t2_original.antsvol(
        data=(ribbon_orig_ants.numpy() > 0.5).astype(np.float32),
        apply_mask=False
    )

    return Volume(
        antsvol=ribbon_original,
        niftiheader=t2_original.header,
        mask=t2_original.mask,
    )


def surface_reconstruction(t2_aligned: Volume, hemi: str, device: str):

    # Set model
    surf_recon_models = ddsp.surface.load(device=device)
    surf_recon_wm = vars(surf_recon_models)[hemi].wm
    surf_recon_pial = vars(surf_recon_models)[hemi].pial

    # Input vertices and faces


    tensor_t2_aligned = torch.Tensor(t2_aligned.data()[None, None]).to(device)
    if hemi == "left":
        tensor_in = tensor_t2_aligned[:, :, 64:]
        vert_left_in = surf_left_in.agg_data('pointset')
        face_left_in = surf_left_in.agg_data('triangle')
        vert_left_in = apply_affine_mat(
            vert_left_in, np.linalg.inv(affine_t2_atlas))
        vert_left_in = vert_left_in - [64, 0, 0]
        face_left_in = face_left_in[:, [2, 1, 0]]
        vert_left_in = torch.Tensor(vert_left_in[None]).to(device)
        face_left_in = torch.LongTensor(face_left_in[None]).to(device)


        # clip the left hemisphere
        vol_in = vol_t2_align[:, :, 64:]
        vert_in = vert_left_in
        face_in = face_left_in

    elif hemi == 'right':




        # clip the right hemisphere
        vol_in = vol_t2_align[:, :, :112]
        vert_in = vert_right_in
        face_in = face_right_in

    # wm and pial surfaces reconstruction
    with torch.no_grad():
        vert_wm = surf_recon_wm(vert_in, vol_in, n_steps=7)
        vert_wm = cot_laplacian_smooth(vert_wm, face_in, n_iters=1)
        vert_pial = surf_recon_pial(vert_wm, vol_in, n_steps=7)
        vert_pial = laplacian_smooth(vert_pial, face_in, n_iters=1)

    # torch.Tensor -> numpy.array
    vert_wm_align = vert_wm[0].cpu().numpy()
    vert_pial_align = vert_pial[0].cpu().numpy()
    face_align = face_in[0].cpu().numpy()

    # transform vertices to original space
    if surf_hemi == 'left':
        # pad the left hemisphere to full brain
        vert_wm_orig = vert_wm_align + [64, 0, 0]
        vert_pial_orig = vert_pial_align + [64, 0, 0]
    elif surf_hemi == 'right':
        vert_wm_orig = vert_wm_align.copy()
        vert_pial_orig = vert_pial_align.copy()
    vert_wm_orig = apply_affine_mat(
        vert_wm_orig, affine_t2_align)
    vert_pial_orig = apply_affine_mat(
        vert_pial_orig, affine_t2_align)
    face_orig = face_align[:, [2, 1, 0]]
    # midthickness surface
    vert_mid_orig = (vert_wm_orig + vert_pial_orig) / 2

    # save as .surf.gii
    save_gifti_surface(
        vert_wm_orig, face_orig,
        save_dir=subj_out_dir + '_hemi-' + surf_hemi + '_wm.surf.gii',
        surf_hemi=surf_hemi, surf_type='wm')
    save_gifti_surface(
        vert_pial_orig, face_orig,
        save_dir=subj_out_dir + '_hemi-' + surf_hemi + '_pial.surf.gii',
        surf_hemi=surf_hemi, surf_type='pial')
    save_gifti_surface(
        vert_mid_orig, face_orig,
        save_dir=subj_out_dir + '_hemi-' + surf_hemi + '_midthickness.surf.gii',
        surf_hemi=surf_hemi, surf_type='midthickness')

        # send to gpu for the following processing
        vert_wm = torch.Tensor(vert_wm_orig).unsqueeze(0).to(device)
        vert_pial = torch.Tensor(vert_pial_orig).unsqueeze(0).to(device)
        vert_mid = torch.Tensor(vert_mid_orig).unsqueeze(0).to(device)
        face = torch.LongTensor(face_orig).unsqueeze(0).to(device)

        t_surf_end = time.time()
        t_surf = t_surf_end - t_surf_start
        logger.info(
            'Surface reconstruction ({}) ends. Runtime: {} sec.'.format(
                surf_hemi, np.round(t_surf, 4)))

        # ------ Surface Inflation ------
        logger.info('----------------------------------------')
        logger.info('Surface inflation ({}) starts ...'.format(surf_hemi))
        t_inflate_start = time.time()

        # create inflated and very_inflated surfaces
        # if device is cpu, use wb_command for inflation (faster)
        if device == 'cpu':
            vert_inflated_orig, vert_vinflated_orig = \
                wb_generate_inflated_surfaces(
                    subj_out_dir, surf_hemi, iter_scale=3.0)
        else:  # cuda acceleration
            vert_inflated, vert_vinflated = generate_inflated_surfaces(
                vert_mid, face, iter_scale=3.0)
            vert_inflated_orig = vert_inflated[0].cpu().numpy()
            vert_vinflated_orig = vert_vinflated[0].cpu().numpy()

        # save as .surf.gii
        save_gifti_surface(
            vert_inflated_orig, face_orig,
            save_dir=subj_out_dir + '_hemi-' + surf_hemi + '_inflated.surf.gii',
            surf_hemi=surf_hemi, surf_type='inflated')
        save_gifti_surface(
            vert_vinflated_orig, face_orig,
            save_dir=subj_out_dir + '_hemi-' + surf_hemi + '_vinflated.surf.gii',
            surf_hemi=surf_hemi, surf_type='vinflated')

        t_inflate_end = time.time()
        t_inflate = t_inflate_end - t_inflate_start
        logger.info('Surface inflation ({}) ends. Runtime: {} sec.'.format(
            surf_hemi, np.round(t_inflate, 4)))


def spherical_projection():

    # Set model, input vertices and faces
    if surf_hemi == 'left':
        sphere_proj = sphere_proj_left
        vert_sphere_in = vert_sphere_left_in
        bc_coord = bc_coord_left
        face_id = face_left_id
    elif surf_hemi == 'right':
        sphere_proj = sphere_proj_right
        vert_sphere_in = vert_sphere_right_in
        bc_coord = bc_coord_right
        face_id = face_right_id

    # interpolate to 160k template
    vert_wm_160k = (vert_wm_orig[face_id] * bc_coord[..., None]).sum(-2)
    vert_wm_160k = torch.Tensor(vert_wm_160k[None]).to(device)
    feat_160k = torch.cat([vert_sphere_160k, vert_wm_160k], dim=-1)

    with torch.no_grad():
        vert_sphere = sphere_proj(
            feat_160k, vert_sphere_in, n_steps=7)

    # compute metric distortion
    edge = torch.cat([
        face[0, :, [0, 1]],
        face[0, :, [1, 2]],
        face[0, :, [2, 0]]], dim=0).T
    edge_distort = 100. * edge_distortion(
        vert_sphere, vert_wm, edge).item()
    area_distort = 100. * area_distortion(
        vert_sphere, vert_wm, face).item()
    logger.info(
        'Edge distortion: {}%'.format(np.round(edge_distort, 2)))
    logger.info(
        'Area distortion: {}%'.format(np.round(area_distort, 2)))

    # save as .surf.gii
    vert_sphere = vert_sphere[0].cpu().numpy()
    save_gifti_surface(
        vert_sphere, face_orig,
        save_dir=subj_out_dir + '_hemi-' + surf_hemi + '_sphere.surf.gii',
        surf_hemi=surf_hemi, surf_type='sphere')

    t_sphere_end = time.time()
    t_sphere = t_sphere_end - t_sphere_start
    logger.info('Spherical mapping ({}) ends. Runtime: {} sec.'.format(
        surf_hemi, np.round(t_sphere, 4)))


def cortical_feature_estimation():

    logger.info('Estimate cortical thickness ...', end=' ')
    thickness = cortical_thickness(vert_wm, vert_pial)
    thickness = metric_dilation(
        torch.Tensor(thickness[None, :, None]).to(device),
        face, n_iters=10)
    save_gifti_metric(
        metric=thickness,
        save_dir=subj_out_dir + '_hemi-' + surf_hemi + '_thickness.shape.gii',
        surf_hemi=surf_hemi, metric_type='thickness')
    logger.info('Done.')

    logger.info('Estimate curvature ...', end=' ')
    curv = curvature(vert_wm, face, smooth_iters=5)
    save_gifti_metric(
        metric=curv,
        save_dir=subj_out_dir + '_hemi-' + surf_hemi + '_curv.shape.gii',
        surf_hemi=surf_hemi, metric_type='curv')
    logger.info('Done.')

    logger.info('Estimate sulcal depth ...', end=' ')
    sulc = sulcal_depth(vert_wm, face, verbose=False)
    save_gifti_metric(
        metric=sulc,
        save_dir=subj_out_dir + '_hemi-' + surf_hemi + '_sulc.shape.gii',
        surf_hemi=surf_hemi, metric_type='sulc')
    logger.info('Done.')

    # estimate myelin map based on
    # t1-to-t2 ratio, midthickness surface,
    # cortical thickness and cortical ribbon

    if t1_exists:
        logger.info('Estimate myelin map ...', end=' ')
        myelin = myelin_map(
            subj_dir=subj_out_dir, surf_hemi=surf_hemi)
        # metric dilation
        myelin = metric_dilation(
            torch.Tensor(myelin[None, :, None]).to(device),
            face, n_iters=10)
        # save myelin map
        save_gifti_metric(
            metric=myelin,
            save_dir=subj_out_dir + '_hemi-' + surf_hemi + '_myelinmap.shape.gii',
            surf_hemi=surf_hemi, metric_type='myelinmap')

        # smooth myelin map
        smoothed_myelin = smooth_myelin_map(
            subj_dir=subj_out_dir, surf_hemi=surf_hemi)
        save_gifti_metric(
            metric=smoothed_myelin,
            save_dir=subj_out_dir + '_hemi-' + surf_hemi + \
                     '_smoothed_myelinmap.shape.gii',
            surf_hemi=surf_hemi,
            metric_type='smoothed_myelinmap')
        logger.info('Done.')

    t_feature_end = time.time()
    t_feature = t_feature_end - t_feature_start
    logger.info('Feature estimation ({}) ends. Runtime: {} sec.'.format(
        surf_hemi, np.round(t_feature, 4)))


def conclude():
    logger.info('----------------------------------------')
    # clean temp data
    os.remove(subj_out_dir + '_rigid_0GenericAffine.mat')
    os.remove(subj_out_dir + '_affine_0GenericAffine.mat')
    os.remove(subj_out_dir + '_ribbon.nii.gz')
    if os.path.exists(subj_out_dir + '_T1wDividedByT2w.nii.gz'):
        os.remove(subj_out_dir + '_T1wDividedByT2w.nii.gz')
    # create .spec file for visualization
    create_wb_spec(subj_out_dir)
    t_end = time.time()
    logger.info('Finished. Total runtime: {} sec.'.format(
        np.round(t_end - t_start, 4)))
    logger.info('========================================')


def create_cli():
    parser = argparse.ArgumentParser(description="dHCP DL Surface Pipeline")
    parser.add_argument(
        '--t2', default=None, type=str,
        help='Suffix of T2 image file.'
    )
    parser.add_argument(
        '--t1', default=None, type=str,
        help='Suffix of T1 image file.'
    )
    parser.add_argument(
        '--mask', default=None, type=str,
        help='Suffix of brain mask file.'
    )
    parser.add_argument(
        '--out_dir', default=None, type=str,
        help='Directory for saving the output of the pipeline.'
    )
    parser.add_argument(
        '--device', default='cuda', type=str,
        help='Device for running the pipeline: [cuda, cpu]'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print debugging information.'
    )
    return parser


def configure_output_directory(args):
    args.out_dir = os.path.abspath(args.out_dir)
    if os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    else:
        os.makedirs(args.out_dir, exist_ok=False)


def create_logger(outputdir, level=LOGLEVEL, mode=LOGMODE):
    """
    Creates a suitably configured Logger instance.

    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers = []  # delete any existing handlers to avoid duplicate logs
    logger.setLevel(1)
    formatter = logging.Formatter(
        fmt='%(asctime)s Process-%(process)d %(levelname)s (%(lineno)d) '
            '- %(message)s',
        datefmt='[%Y-%m-%d %H:%M:%S]')

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
if __name__ == '__main__':
    parser = create_cli()
    if len(sys.argv) > 1:
        args = parser.parse_args()
        configure_output_directory(args)
        logger = create_logger(args.out_dir)
        main(args)
    else:
        parser.print_help()
