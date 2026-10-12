"""Show readable time estimates for the running frozen P2 probe."""

import argparse
import json
from pathlib import Path
import re
import time


def duration(seconds):
    minutes = max(0, round(seconds / 60))
    return '{} h {:02d} min'.format(minutes // 60, minutes % 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=Path, default=Path(
        'logs/p2_query_detail_frozen_probe_20261012_bs1_nohup.log'))
    args = parser.parse_args()
    content = args.log.read_text(errors='replace')
    status_file = args.log.with_name(args.log.name.replace('_nohup.log', '.status'))
    if status_file.exists():
        print(status_file.read_text().strip())
        return
    matches = list(re.finditer(
        r'\[FrozenProbe\] epoch=(\d+)/(\d+) step=(\d+)/(\d+).*?eta_seconds=(\d+)',
        content))
    if not matches:
        print('Preparing model / evaluating the initial baseline.')
        return
    epoch, epochs, step, steps, eta = map(int, matches[-1].groups())
    last_update_age = max(0, time.time() - args.log.stat().st_mtime)
    current_remaining = max(0, eta - last_update_age)
    print('Epoch: {}/{} | Step: {}/{}'.format(epoch, epochs, step, steps))
    print('Current epoch remaining: {}'.format(duration(current_remaining)))
    baseline_matches = re.findall(r'\[FrozenProbeBaseline\] (\{[^\n]+\})', content)
    evaluation_seconds = json.loads(baseline_matches[-1])['seconds'] if baseline_matches else 75.
    average_step = eta / max(1, steps - step - 1)
    total_remaining = (current_remaining + (epochs - epoch) * steps * average_step +
                       (epochs - epoch + 1) * evaluation_seconds)
    print('All epochs + evaluations remaining: approximately {}'.format(
        duration(total_remaining)))
    print('Estimates update with training speed; no fixed completion time.')


if __name__ == '__main__':
    main()
