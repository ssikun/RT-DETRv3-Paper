# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys

import paddle.distributed as dist

__all__ = ['setup_logger', 'set_logger_output']

logger_initialized = []
_default_output = None
_default_log_ranks = '0'


def _parse_log_ranks(log_ranks):
    if isinstance(log_ranks, str):
        return [int(i) for i in log_ranks.split(',')]
    if isinstance(log_ranks, int):
        return [log_ranks]
    return log_ranks


def _build_formatter():
    return logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S")


def _log_filename(output, local_rank):
    if output.endswith(".txt") or output.endswith(".log"):
        filename = output
    else:
        filename = os.path.join(output, "log.txt")
    if local_rank > 0:
        filename = filename + ".rank{}".format(local_rank)
    return os.path.abspath(filename)


def _add_file_handler(logger, output, formatter, log_ranks='0'):
    local_rank = dist.get_rank()
    if local_rank not in _parse_log_ranks(log_ranks):
        return

    filename = _log_filename(output, local_rank)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and \
                getattr(handler, 'baseFilename', None) == filename:
            return

    log_dir = os.path.dirname(filename)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(filename, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def set_logger_output(output, logger_names=None, log_ranks='0'):
    global _default_output, _default_log_ranks
    _default_output = output
    _default_log_ranks = log_ranks

    formatter = _build_formatter()
    if logger_names is None:
        logger_names = list(logger_initialized)
    for name in logger_names:
        _add_file_handler(logging.getLogger(name), output, formatter, log_ranks)


def setup_logger(name="ppdet", output=None, log_ranks="0"):
    """
    Initialize logger and set its verbosity level to INFO.
    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        log_ranks (str): The ids of gpu to log which are separated by "," when more than 1, "0" by default.

    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    use_default_output = output is None and _default_output is not None
    output = output if output is not None else _default_output
    if use_default_output:
        log_ranks = _default_log_ranks
    formatter = _build_formatter()

    if name in logger_initialized:
        if output is not None:
            _add_file_handler(logger, output, formatter, log_ranks)
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_ranks = _parse_log_ranks(log_ranks)

    # stdout logging: master only
    local_rank = dist.get_rank()
    if local_rank in log_ranks:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: configured ranks only
    if output is not None:
        _add_file_handler(logger, output, formatter, log_ranks)

    logger_initialized.append(name)
    return logger

