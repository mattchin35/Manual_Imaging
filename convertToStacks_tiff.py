"""Convert Prairie-generated individual TIFF files to hdf5 stacks"""
from __future__ import print_function
import os
from os.path import join, exists, split
import shutil
import argparse
import re
import fileinput
import sys
from xml.etree import ElementTree
import numpy as np
from distutils.version import LooseVersion
from datetime import datetime as dt
from datetime import timedelta
from PIL import Image
from uuid import uuid4
import h5py
import itertools as it
import fcntl
import traceback
from distutils.util import strtobool

FAIL_FILE = 'convert_fail'


def touch(path):
    with open(path, 'a'):
        os.utime(path, None)


def timestamp():
    return dt.strftime(dt.now(), '[%Y-%m-%d %H:%M:%S]')


def parse_cfg_file(cfg_filepath, version=None):
    """Parses the prairie config file and determines the number of iterations
    and experiment type for each sequence"""
    try:
        cfg_tree = ElementTree.parse(cfg_filepath)
        # Shouldn't need to except the TypeError, but something is failing
        # within the ElementTree parser
    except (ElementTree.ParseError, TypeError):
        reformat_prairie_cfg(cfg_filepath)
        cfg_tree = ElementTree.parse(cfg_filepath)

    tSeries_element = cfg_tree.find('TSeries')
    if version == LooseVersion('4.3.2.24'):
        n_iterations = int(tSeries_element.find(
            'PVTSeriesElementSequenceTimed').get('repetitions'))
    else:
        try:
            n_iterations = int(tSeries_element.get('iterations'))
        except TypeError:
            n_iterations = None

    elements = []
    if version == LooseVersion('4.3.2.24'):
        for elem in tSeries_element:
            if 'PVTSeriesElementSequence' in elem.tag:
                elements.append(
                    ('PVTSeriesElementSequence', int(elem.get('repetitions'))))
    else:
        for elem in tSeries_element:
            if elem.tag in ['PVTSeriesElementSequence',
                            'PVTSeriesElementZSeries']:
                elements.append((elem.tag, int(elem.get('repetitions'))))

    return elements, n_iterations


def get_element_size_um(xml_filepath, prairie_version):
    """Determine the size in um of x and y in order to store it with the
    data. The HDF5 plugin for ImageJ will read this metadata"""
    if prairie_version >= LooseVersion('5.2'):
        for _, elem in ElementTree.iterparse(xml_filepath):
            if elem.get('key') == 'micronsPerPixel':
                for value in elem.findall('IndexedValue'):
                    if value.get('index') == "XAxis":
                        x = float(value.get('value'))
                    elif value.get('index') == "YAxis":
                        y = float(value.get('value'))
                return (1, y, x)
    else:
        for _, elem in ElementTree.iterparse(xml_filepath):
            if elem.tag == 'PVStateShard':
                for key in elem.findall('Key'):
                    if key.get('key') == 'micronsPerPixel_XAxis':
                        x = float(key.get('value'))
                    elif key.get('key') == 'micronsPerPixel_YAxis':
                        y = float(key.get('value'))
                return (1, y, x)
    print('Unable to identify element size, returning default value')
    return (1, 1, 1)


def reformat_prairie_cfg(cfg_filepath):
    """Replace newline symbols in .cfg with a space"""
    for line in fileinput.input(cfg_filepath, inplace=1):
        if '&#x1;' in line:
            line = line.replace('&#x1;', ' ')
        sys.stdout.write(line)


def get_prairieview_version(xml_filepath):
    """Return Prairieview version number"""
    for _, elem in ElementTree.iterparse(xml_filepath, events=("start",)):
        if elem.tag == 'PVScan':
            return LooseVersion(elem.get('version'))


def save_multipage_TIFF(input_filenames, output_filename):
    """Read in a list of input filenames and write out as a multi-page TIFF"""
    # raise NotImplemented("Need to update for multiple planes")
    from libtiff import TIFF
    import tifffile as tff
    f = TIFF.open(input_filenames[0][0][0], 'r')
    first_img = f.read_image()
    f.close()

    output_array = np.empty(
        [len(input_filenames), first_img.shape[0], first_img.shape[1]],
        dtype=first_img.dtype)
    for idx, filename in enumerate(input_filenames):
        f = TIFF.open(filename[0][0], 'r')
        output_array[idx, :, :] = f.read_image()
        f.close()

    tff.imwrite(output_filename, output_array)
    #f = TIFF.open(output_filename, 'w')
    #f.write_image(output_array)
    #f.close()


def save_HDF5(input_filenames, output_filename, channel_names=None,
              element_size_um=(1, 1, 1), group='/', key='imaging',
              temp_dir=None, compression=None, skip_bad_files=False):
    """Read in a list of input filenames and write out as an HDF5 dataset

    Parameters
    ----------
    input_filenames : list of list of list of strs
        Filenames of files to convert, in tzc (time, z-plane, channel) order.
    output_filename : str
        Output filename to write the HDF5 file to.
    group : str, optional
        The HDF5 group containing the imaging data.
        Defaults to using the root group '/'.
    key : str, optional
        The key for indexing the the HDF5 dataset containing the imaging data.
        Defaults to 'imaging'
    channel_names : list of strs, optional
        Labels for channels. Defaults to numerical indexes.
    element_size_um : 3-element tuple, optional
        Size of each pixel in um. Defaults to (1, 1, 1).
    temp_dir : str, optional
        If not None, move the TIFFs to 'temp_dir' before opening, allows for
        separating read/write operations on different physical disks.
    compression : optional, None or string
        Compression argument to pass to h5py
    skip_bad_files: bool, optional
        If True, skips bad files, storing a 2D-array (num_bad_files, 3) in
        group/bad_frames. The columns correspond to the t, z, c indices of each
        bad frame.
    
    """

    name = join(group, key)

    # Load the first good image we can find, fail if all bad
    file_found = False
    for frame in input_filenames:
        for plane in frame:
            for channel in plane:
                try:
                    f = Image.open(input_filenames[0][0][0], 'r')
                except IOError:
                    pass
                else:
                    print(f.size)
                    # print(np.array(f.getdata()).shape)
                    print(np.array(f.load()))
                    first_img = np.array(f)
                    file_found = True
                    break
            if file_found:
                break
        if file_found:
            break
    if not file_found:
        raise IOError("No good files found: {}".format(output_filename))

    if temp_dir:
        try:
            temp_path = os.path.join(temp_dir, str(uuid4()))
            os.mkdir(temp_path)
            temp_filenames = []
            for frame in input_filenames:
                temp_filenames.append([])
                for z_plane in frame:
                    temp_filenames[-1].append([])
                    for filename in z_plane:
                        temp_filename = os.path.join(
                            temp_path, os.path.basename(filename))
                        shutil.copyfile(filename, temp_filename)
                        temp_filenames[-1][-1].append(temp_filename)
            input_filenames = temp_filenames
        except:
            shutil.rmtree(temp_path)
            raise

    # Order is t, z, y, x, c
    # print(input_filenames)
    # print(first_img)
    output_shape = (len(input_filenames), len(input_filenames[0]),
                    first_img.shape[0], first_img.shape[1],
                    len(input_filenames[0][0]))

    bad_frames = []

    h5 = h5py.File(output_filename, 'w', libver='latest')
    h5[group].create_dataset(
        key, output_shape, first_img.dtype, maxshape=output_shape,
        chunks=(1, 1, output_shape[2], output_shape[3], 1),
        compression=compression)
    try:
        for frame_idx, frame in it.izip(it.count(), input_filenames):
            for z_idx, z_plane in it.izip(it.count(), frame):
                for ch_idx, filename in it.izip(it.count(), z_plane):
                    try:
                        f = Image.open(filename, 'r')
                    except:
                        if skip_bad_files:
                            print("Bad file: " + filename)
                            bad_frames.append((frame_idx, z_idx, ch_idx))
                            f_data = np.zeros(
                                output_shape[2:4], dtype=first_img.dtype)
                        else:
                            raise
                    else:
                        f_data = np.array(f)
                        f.close()
                    h5[name][frame_idx, z_idx, :, :, ch_idx] = f_data

        for idx, label in enumerate(['t', 'z', 'y', 'x', 'c']):
            h5[name].dims[idx].label = label
        if channel_names is None:
            channel_names = np.arange(output_shape[4])
        h5[name].attrs['channel_names'] = np.array(channel_names)
        h5[name].attrs['element_size_um'] = np.array(element_size_um)
        if skip_bad_files and len(bad_frames):
            bad_frames_name = join(group, 'bad_frames')
            h5[bad_frames_name] = np.vstack(bad_frames)
        h5.close()
    except:
        # If anything fails, delete the incomplete file
        h5.close()
        os.remove(output_filename)
        if temp_dir:
            shutil.rmtree(temp_path)
        raise

    # Verify the integrity of the saved file
    try:
        h5 = h5py.File(output_filename, 'r')
        for frame_idx, frame in it.izip(it.count(), input_filenames):
            for z_idx, z_plane in it.izip(it.count(), frame):
                for ch_idx, filename in it.izip(it.count(), z_plane):
                    if (frame_idx, z_idx, ch_idx) in bad_frames:
                        assert(np.all(h5[name][frame_idx, z_idx, :, :, ch_idx]
                               == 0))
                    else:
                        f = Image.open(filename, 'r')
                        assert(np.all(h5[name][frame_idx, z_idx, :, :, ch_idx]
                               == np.array(f)))
                        f.close()
        h5.close()
    except:
        # If the check failed, delete the bad file
        h5.close()
        os.remove(output_filename)
        raise
    finally:
        if temp_dir:
            shutil.rmtree(temp_path)


def convert_to_HDF5(
        directory=os.curdir, overwrite=False, no_action=False, delete=False,
        temp_dir=None, move_dir=None, debug=False, force=False,
        compression=None, skip_bad_files=False, tiff_output=False):
    """Function to convert all found TIFF files to HDF5.
    Based on Prairie's naming convention, parses their xml file to extract
    filenames.

    Arguments
    ---------
    directory -- path to walk down to find data
    overwrite -- If True, overwrites h5 file if tiffs are still there
    no_action -- If True, do nothing, just report messages
    delete -- If True, delete tiffs when done
    temp_dir -- If not None, copies all TIFFS to temp_dir and reads from there
        during the conversion. Try using '/mnt/backup/data/tmp'
    move_dir -- If not None, after successful completion of conversions, move
        the parent directory of the h5 files to 'move_dir', mirroring the
        relative path from 'directory'.
    debug -- If don't suppress errors and fail on the first exception.
    force -- If true, ignore fail file
    compression -- optional, compression argument to pass to h5py
    skip_bad_files -- optional, replaces bad TIFF files with all zeros.

    Ex. convert_to_HDF5('/scratch/data/', move_dir='/data') will move
    /scratch/data/Jeff/mouse1 to /data/Jeff/mouse1 upon successful completion.
    
    """

    group = '/'
    key = 'imaging'
    print(directory)

    file_check_regex = re.compile('\S_Cycle.*Ch[12]_0+\d+.*tif')
    # channel_regex = re.compile('.*(Ch\d+)_.*tif')
    # cycle_regex = re.compile('.*Cycle0+(\d*)_.*tif')
    
    for cur_dir, folders, files in os.walk(directory):

        # Assemble a list of all .tif files
        tif_files = [f for f in files if re.search(file_check_regex, f)]
        print(len(tif_files))
        if len(tif_files) < 3:
            print('quitting, no tiff files')
            # Nothing to do here...
            continue

        fail_file = join(cur_dir, FAIL_FILE)
        # print(sorted(
        #     [dt.fromtimestamp(os.path.getctime(join(cur_dir, f)))
        #      for f in tif_files])

        newest_tif_time = sorted(
            [dt.fromtimestamp(os.path.getctime(join(cur_dir, f)))
             for f in tif_files])[-1]

        # Check to make sure that we are not currently transferring files
        now = dt.today()
        threshold = timedelta(seconds=120)
        if now - newest_tif_time < threshold:
            print("Files too new: ", cur_dir)
            continue

        if not force and os.path.exists(fail_file):
            fail_time = dt.fromtimestamp(os.path.getmtime(fail_file))
            if fail_time > newest_tif_time:
                # Quietly skip directory
                print("fail file present")
                continue

        match = re.search(file_check_regex, tif_files[0])
        basename = tif_files[0][:match.start() + 1]
        xml_filename = basename + '.xml'

        try:
            lockfile = open(join(cur_dir, xml_filename))
        except IOError:
            print("{} Unable to locate XML file: {}".format(
                timestamp(), cur_dir))
            with open(fail_file, 'w') as f:
                traceback.print_exc(file=f)
            continue

        # Lock the xml file, if it can't be locked just continue
        # This will hold the lock if it exits early before conversion,
        # but it will be released again when the process ends
        """try:
            fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            print("cannot access xml file")
            continue"""

        # Parse the xml name, Prairie version, and cfg filename
        try:
            version = get_prairieview_version(join(cur_dir, xml_filename))
        except (IOError, ElementTree.ParseError):
            print("{} XML Parse error: {}".format(timestamp(), cur_dir))
            with open(fail_file, 'w') as f:
                traceback.print_exc(file=f)
            continue

        # If data was recorded pre-5.0, just skip the folder
        # if version < LooseVersion('5.0'):
        #     failed = "Prairie version too old ({}): {}".format(
        #         version, cur_dir)
        #     print failed
        #     with open(fail_file, 'w') as f:
        #         f.write(failed + '\n')
        #     continue

        if version > LooseVersion('5.2'):
            cfg_filename = xml_filename.replace('.xml', '.env')
        else:
            cfg_filename = xml_filename.replace('.xml', 'Config.cfg')

        protocol_elements, n_cycles = parse_cfg_file(
            join(cur_dir, cfg_filename), version)

        # Older Prairie versions don't actually store iterations/cycles
        # anywhere...hack to not need to know beforehand
        if n_cycles is None:
            cycles = it.count()
        else:
            cycles = range(n_cycles)

        # Create a generator of all the Sequences in the XML
        sequences = (
            elem for _, elem in
            ElementTree.iterparse(join(cur_dir, xml_filename))
            if elem.tag == 'Sequence' and elem.get('type') !=
            'TSeries Voltage Output Experiment')

        # Iterate over protocols and cycles
        # There will be one HDF5 file per cycle-protocol
        failed = False
        iter_break = False
        tiffs_to_save = {}
        for cycle in cycles:
            for idx, (protocol, reps) in enumerate(protocol_elements):
                if tiff_output:
                    ext = 'tif'
                else:
                    ext = 'h5'
                output_filename = '{}_Cycle{:05d}_Element{:05d}.'.format(
                    basename, cycle + 1, idx + 1) + ext

                if protocol == 'PVTSeriesElementSequence':
                    # This will both check for mis-matched number of
                    # sequences and fix for old Prairie not knowing cycles
                    try:
                        sequence = next(sequences)
                        # sequence = sequences.next()  # python2
                    except StopIteration:
                        if version < LooseVersion('5.2'):
                            iter_break = True
                            break
                        else:
                            err_msg = '{} Sequence length mis-match, '.format(
                                timestamp()) + \
                                '{} expected, {} actual: {}'.format(
                                    len(protocol_elements), idx + 1, cur_dir)
                            print(err_msg)
                            failed = err_msg
                            if debug:
                                raise
                            break
                    frames = sequence.findall('Frame')
                    channels = [ff.get('channelName')
                                for ff in frames[0].findall('File')]
                    channels.sort()
                    if len(frames) != reps:
                        err_msg = '{} Frame/rep '.format(timestamp()) \
                            + 'mismatch, {} frames, {} reps: {}'.format(
                                len(frames), reps, cur_dir)
                        print(err_msg)
                        failed = err_msg
                        break
                    tiff_files = []
                    # Each frame is a time step
                    for frame in frames:
                        files = [join(cur_dir, ff.get('filename'))
                                 for ff in frame.findall('File')]
                        files.sort()
                        tiff_files.append([files])

                elif protocol == 'PVTSeriesElementZSeries':
                    tiff_files = []
                    # Each sequence/rep is a time step
                    channels = None
                    for rep in range(reps):
                        try:
                            sequence = sequences.next()
                        except StopIteration:
                            # If we run out of sequences on the first rep,
                            # there's just no more cycles left, which is fine.
                            # If happens in the middle of reps we are actually
                            # missing data.
                            if version < LooseVersion('5.2') and rep == 0:
                                iter_break = True
                                break
                            else:
                                err_msg = \
                                    '{} Sequence length mis-match, '.format(
                                        timestamp()) \
                                    + '{} expected, {} actual: {}'.format(
                                        reps, rep + 1, cur_dir)
                                print(err_msg)
                                failed = err_msg
                                if debug:
                                    raise
                                break
                        frames = sequence.findall('Frame')
                        if channels is None:
                            channels = [ff.get('channelName')
                                        for ff in frames[0].findall('File')]
                            channels.sort()
                        tiff_files.append([])
                        # Each frame is a z-plane
                        for frame in frames:
                            files = [join(cur_dir, ff.get('filename'))
                                     for ff in frame.findall('File')]
                            files.sort()
                            tiff_files[-1].append(files)
                    if failed or iter_break:
                        break
                else:
                    err_msg = '{} Unrecognized '.format(timestamp()) + \
                        'protocol element, skipping directory: {}, {}'.format(
                            cur_dir, protocol)
                    print(err_msg)
                    failed = err_msg
                    if debug:
                        raise Exception
                    break

                # Only add new h5 files to the convert list, but we still
                # need to iterate over all elements, so this is at the end
                if overwrite or not exists(join(cur_dir, output_filename)):
                    tiffs_to_save[output_filename] = (tiff_files, channels)

            if failed or iter_break:
                break

        if failed:
            if len(failed) == 3:
                with open(fail_file, 'w') as f:
                    traceback.print_exception(
                        failed[0], failed[1], failed[2], file=f)
            else:
                with open(fail_file, 'w') as f:
                    f.write(failed + '\n')
            continue

        # If there's nothing to do, just continue
        if not len(tiffs_to_save):
            continue

        # Make sure we've exactly gone through all of the sequences
        try:
            next(sequences)
            # sequences.next()
        except StopIteration:
            pass
        else:
            err_msg = '{} Sequence length mis-matching'.format(timestamp()) + \
                ', skipping directory: {}'.format(cur_dir)
            print(err_msg)
            with open(fail_file, 'w') as f:
                f.write(err_msg + '\n')
            if debug:
                raise Exception
            continue

        # Check to make sure all of the files are there
        # for files, _ in tiffs_to_save.itervalues(): # python2
        for files, _ in iter(tiffs_to_save.values()):
            for frame in files:
                for z_plane in frame:
                    for f in z_plane:
                        if not exists(f):
                            err_msg = '{} Missing file'.format(timestamp()) + \
                                ', skipping directory: {}'.format(f)
                            print(err_msg)
                            failed = err_msg
                            if debug:
                                raise Exception
                            break
                    if failed:
                        break
                if failed:
                    break
            if failed:
                break
        if failed:
            with open(fail_file, 'w') as f:
                f.write(failed + '\n')
            continue

        # Get the size in um of the x and y dimensions
        element_size_um = get_element_size_um(
            join(cur_dir, xml_filename), version)

        failed = False
        for  n, (tiffs, channels) in tiffs_to_save.iteritems():
            print("{} Creating {}".format(
                timestamp(), join(cur_dir, output_filename)))
            if not no_action:
                try:
                    flat_tiffs = [z for x in tiffs for y in x for z in y]
                    hd, tl = os.path.split(flat_tiffs[0])
                    OME_fld = join(hd, "OME-TIFF")
                    if not os.path.exists(OME_fld):
                        os.mkdir(OME_fld)

                    if tiff_output:
                        save_multipage_TIFF(tiffs, join(cur_dir, output_filename))
                    else:
                        save_HDF5(
                            tiffs, join(cur_dir, output_filename),
                            channel_names=channels, group=group, key=key,
                            element_size_um=element_size_um,
                            compression=compression, temp_dir=temp_dir,
                            skip_bad_files=skip_bad_files)

                    [shutil.move(t, OME_fld) for t in flat_tiffs]

                except:
                    print("{} FAILED creating {}".format(
                          timestamp(), output_filename))
                    failed = sys.exc_info()
                    if debug:
                        raise
                else:
                    if delete:
                        print("{} Successfully created ".format(timestamp()) +
                              "{}, deleting original files".format(
                                  output_filename))
                        if skip_bad_files:
                            h5_filename = join(cur_dir, output_filename)
                            h5_file = h5py.File(h5_filename, 'r')
                            bad_files_key = join(group, 'bad_frames')
                            if bad_files_key in h5_file:
                                n_bad_files = h5_file[bad_files_key].shape[0]
                            else:
                                n_bad_files = 0
                            if n_bad_files:
                                try:
                                    prompt = strtobool(raw_input(
                                        "{} bad file(s) found. ".format(
                                            n_bad_files) +
                                        "Delete all original files? "))
                                except ValueError:
                                    prompt = False
                                if not prompt:
                                    continue
                        for frame in tiffs_to_save[output_filename][0]:
                            for z_plane in frame:
                                for filename in z_plane:
                                    os.remove(filename)
                    else:
                        print("{} Successfully created {}".format(
                            timestamp(), output_filename))

        if failed:
            with open(fail_file, 'w') as f:
                traceback.print_exception(
                    failed[0], failed[1], failed[2], file=f)
        else:
            try:
                os.remove(fail_file)
            except OSError:
                pass

        if not failed and move_dir:
            rel_path = os.path.relpath(cur_dir, directory)
            new_path = os.path.join(move_dir, rel_path)
            if os.path.isdir(new_path):
                print("{} FAILED moving {} to {}: path already exists".format(
                    timestamp(), cur_dir, new_path))
                if debug:
                    raise Exception
            else:
                try:
                    shutil.move(cur_dir, new_path)
                except:
                    print("{} FAILED moving {} to {}".format(
                          timestamp(), cur_dir, new_path))
                    if debug:
                        raise
                else:
                    print("{} Successfully moved {} to {}".format(
                        timestamp(), cur_dir, new_path))

        # Release the file lock
        lockfile.close()

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "directory", action="store", type=str, default='',
        help="Process any experiment that has a tSeriesDirectory containing \
              'directory'")
    argParser.add_argument(
        "-n", "--no_action", action="store_true",
        help="Do nothing, just report the changes that would be made")
    argParser.add_argument(
        "--delete", action="store_true",
        help="If TIFF stack is successfully created, delete original TIFFs")
    argParser.add_argument(
        "-o", "--overwrite", action="store_true",
        help="Overwrite file if it already exists")
    argParser.add_argument(
        "-t", "--temp_dir", action="store", type=str, default=None,
        help="Copies all data to temp_dir before converting, "
        + "allows for read/write from separate physical disks")
    argParser.add_argument(
        "-m", "--move_dir", action="store", type=str, default=None,
        help="When complete move entire contents of parent directory to new" +
        "path. The relative path from 'directory' will be mirrored to " +
        "'move_dir'.")
    argParser.add_argument(
        "-c", "--compression", action="store", type=str, default=None,
        help="Algorithm to use to compress hdf5. See h5py docs for options.")
    argParser.add_argument(
        "--debug", action="store_true")
    argParser.add_argument(
        "-f", "--force", action="store_true",
        help="Force conversion to run on all data, ignoring fail files")
    argParser.add_argument(
        "--skip", action="store_true",
        help="Skips bad TIFF files, replacing with zeros. Not-scriptable " +
        "with --delete.")
    argParser.add_argument(
        "-tif", "--tiff_output", action="store_true",
        help="Output to tiff files instead of h5 files.")
    args = argParser.parse_args()

    convert_to_HDF5(
        directory=args.directory, overwrite=args.overwrite,
        no_action=args.no_action, delete=args.delete, temp_dir=args.temp_dir,
        move_dir=args.move_dir, debug=args.debug, force=args.force,
        compression=args.compression, skip_bad_files=args.skip, tiff_output=args.tiff_output)
