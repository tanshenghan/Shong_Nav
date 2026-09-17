"""Validate the saved 470x280 R2R RVQ-10/FM-30 collection without rendering."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from data_pipeline.pointing import (  # noqa: E402
    PointingConfig,
    attach_pixel_grid,
    ground_point,
    pixel_grid_arrays,
    pixel_grid_metadata,
    pointing_token,
    terminal_ground_point,
)


def validate(root):
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('format') == 'beta_nav_pointing_v1':
        from data_pipeline.validate_pointing import validate_pointing

        return validate_pointing(root)
    K = np.asarray(manifest['camera']['K'])
    totals = dict(episodes=0, frames=0, full_rvq_10=0, full_fm_30=0, stop_samples=0)
    projection_errors = []
    turns_checked = 0
    for folder in sorted((root/'episodes').iterdir()):
        episode = json.loads((folder/'episode.json').read_text())
        assert episode['complete']
        frames = [json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
        samples = [json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
        data = np.load(folder/'targets.npz', allow_pickle=False)
        n = len(frames)
        assert len(samples) == n == episode['frames']
        assert data['fm_clean_se2_diff'].shape == (n,30,3)
        expected = np.arange(n)[:,None] + np.arange(1,31) < n
        np.testing.assert_array_equal(data['fm_valid_mask'], expected)
        assert data['rvq_valid_mask'].sum() == max(0,n-10)+1
        assert not data['fm_valid_mask'][-1].any()
        assert samples[-1]['explicit_stop'] and samples[-1]['stop_loss_valid']
        for i, frame in enumerate(frames):
            with Image.open(folder/frame['rgb']) as image:
                assert image.size == (470,280) and image.mode == 'RGB'
                image.verify()
            depth = np.load(folder/frame['depth'], allow_pickle=False)
            assert depth['depth_m'].shape == (280,470)
            assert depth['depth_m'].dtype == np.float32
            np.testing.assert_array_equal(depth['valid'], (depth['depth_m']>0)&(depth['depth_m']<10))
            body = np.asarray(frame['world_from_body'])
            cam = np.asarray(frame['world_from_camera_cv'])
            extrinsic = np.linalg.inv(body) @ cam
            np.testing.assert_allclose(extrinsic[:3,3],[0,.88,0],atol=1e-5)
            np.testing.assert_allclose(extrinsic[:3,:3],np.diag([1,-1,-1]),atol=1e-5)
            if manifest.get('format') == 'beta_nav_v2':
                assert 'next_expert_action' not in frame
                config = PointingConfig.from_metadata(episode['pointing_config'])
                if 'opos_visibility_start_frame' in episode:
                    assert episode['opos_visibility_start_frame'] == config.opos_visibility_start_frame(n)
                apos = ground_point(data['world_from_body'][:,:3,3],i,cam,K,
                                    depth['depth_m'],depth['valid'],config)
                if 'pointing_grid' in episode:
                    h, w = depth['depth_m'].shape
                    assert episode['pointing_grid'] == pixel_grid_metadata(w, h)
                    apos = attach_pixel_grid(apos, w, h)
                    for name, expected_grid in pixel_grid_arrays([apos], 'apos').items():
                        np.testing.assert_array_equal(data[name][i], expected_grid[0])
                assert samples[i]['apos'] == apos
                assert bool(data['apos_valid_mask'][i]) == apos['loss_valid']
                assert int(data['apos_token_id'][i]) == apos['token_id']
                assert apos['token_id'] == config.unknown_token_id or 1 <= apos['token_id'] <= 1296
                assert samples[i]['apos_tokens'] == pointing_token('apos', apos)
                if config.tail_policy == 'terminal':
                    end = min(i + config.horizon, n - 1)
                    assert data['apos_target_frame_id'][i] == end
                    np.testing.assert_allclose(data['apos_target_world'][i],
                                               data['world_from_body'][end,:3,3],atol=1e-8)
                if apos['loss_valid']:
                    np.testing.assert_allclose(data['apos_pixel'][i],apos['pixel'],atol=1e-5)
                if config.opos_mode == 'terminal_ground':
                    h, w = depth['depth_m'].shape
                    opos = attach_pixel_grid(terminal_ground_point(data['world_from_body'][:, :3, 3], cam, K,
                                             depth['depth_m'], depth['valid'], config, frame_index=i), w, h)
                    assert samples[i]['opos'] == opos
                    assert bool(data['opos_valid_mask'][i]) == opos['loss_valid']
                    assert int(data['opos_token_id'][i]) == opos['token_id']
                    assert samples[i]['opos_tokens'] == pointing_token('opos', opos)
                    assert data['opos_target_frame_id'][i] == n - 1
                    np.testing.assert_array_equal(data['opos_target_world'][i],data['world_from_body'][-1,:3,3])
                    for name, values in pixel_grid_arrays([opos], 'opos').items():
                        np.testing.assert_array_equal(data[name][i], values[0])
                    if opos['state'] == 'point':
                        np.testing.assert_allclose(data['opos_pixel'][i],opos['pixel'],atol=1e-4)
                        np.testing.assert_allclose(data['opos_world'][i],opos['world_point'],atol=1e-4)
                    else:
                        assert np.isnan(data['opos_pixel'][i]).all() and np.isnan(data['opos_world'][i]).all()
            action = frame.get('next_expert_action')
            delta = data['fm_clean_se2_diff'][i,0]
            stationary_turn = i+1<n and np.linalg.norm(delta[:2]) < 1e-5 and abs(delta[2]) > 1e-5
            if stationary_turn:
                np.testing.assert_allclose(delta[:2],0,atol=1e-5)
                np.testing.assert_allclose(abs(delta[2]),np.deg2rad(15),atol=1e-5)
                if action in (2,3):
                    expected_yaw = np.deg2rad(15 if action==2 else -15)
                    np.testing.assert_allclose(delta[2],expected_yaw,atol=1e-5)
                    turns_checked += 1
                if len(projection_errors)<50:
                    v,u = np.mgrid[20:260:16,20:450:16]
                    uv = np.stack([u.ravel()+.5,v.ravel()+.5,np.ones(u.size)],axis=1)
                    z = depth['depth_m'][v.ravel(),u.ravel()]
                    good = depth['valid'][v.ravel(),u.ravel()]
                    pc = (np.linalg.inv(K) @ uv.T).T * z[:,None]
                    nextcam = np.asarray(frames[i+1]['world_from_camera_cv'])
                    transform = np.linalg.inv(nextcam) @ cam
                    pc2 = pc @ transform[:3,:3].T + transform[:3,3]
                    projected = pc2 @ K.T
                    uv2 = projected[:,:2]/np.maximum(projected[:,2:],1e-8)
                    good &= (pc2[:,2]>.1)&(uv2[:,0]>=0)&(uv2[:,0]<470)&(uv2[:,1]>=0)&(uv2[:,1]<280)
                    pix = np.floor(uv2[good]).astype(int)
                    other = np.load(folder/frames[i+1]['depth'])
                    valid2 = other['valid'][pix[:,1],pix[:,0]]
                    residual = np.abs(other['depth_m'][pix[:,1],pix[:,0]]-pc2[good,2])
                    if valid2.sum()>20:
                        projection_errors.append(float(np.median(residual[valid2])))
        totals['episodes']+=1
        totals['frames']+=n
        totals['full_rvq_10']+=int(expected[:,:10].all(1).sum())
        totals['full_fm_30']+=int(expected.all(1).sum())
        totals['stop_samples']+=1
    totals['depth_reprojection_turn_pairs']=len(projection_errors)
    totals['median_pair_depth_reprojection_error_m']=float(np.median(projection_errors)) if projection_errors else None
    totals['max_pair_median_depth_reprojection_error_m']=float(max(projection_errors)) if projection_errors else None
    totals['all_saved_images_depths_poses_masks_checked']=True
    totals['all_recorded_turns_match_15_degree_sign']=True if turns_checked else None
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Collected R2R run directory")
    parser.add_argument("--report", type=Path, help="Also write JSON to a NEW report file")
    args = parser.parse_args()
    if args.report is not None and args.report.exists():
        parser.error("report already exists; choose a new filename")
    report = validate(args.dataset)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    print(encoded, end="")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            stream.write(encoded)


if __name__ == "__main__":
    main()
