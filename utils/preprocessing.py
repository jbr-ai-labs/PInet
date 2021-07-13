import sys
# sys.path.append('/home/ant/data/PInet/')
# print(sys.path)
from typing import Optional
from pymol import cmd
from tqdm import tqdm
import os
import logging
import re
import torch
import time
from dx2feature import *
from getResiLabel import *
from sklearn import neighbors
from pinet.model import PointNetDenseCls12




# pdb to wrl
def pdb_to_wrl(directory: str, pdbfile: str, pdbid: Optional[str], output_path: str, suffix: str) -> None:
    """
    Uses pymol to get wrl file (from pdb file): points on the surface format
    @param directory: path to PDB files
    @param pdbfile: PDB file name
    @param pdbid: structure id (four symbols)
    @param output_path: path to output files
    @return: None
    """
    if pdbid is None:
        pdbid = os.path.basename(pdbfile)[:4]
    assert len(pdbid) == 4
    assert suffix in ['l', 'r']

    # clears all objects and resets all parameters to defaults
    cmd.reinitialize()

    cmd.load(os.path.join(directory, pdbfile))
    cmd.set('surface_quality', '0')
    cmd.show_as('surface', 'all')
    cmd.set_view('1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,300,1')
    cmd.save(f"{output_path}wrl/{pdbid}-{suffix}.wrl")
    cmd.delete('all')


def get_coord(wrlfile: str) -> np.ndarray:
    """
    Return surface coordinates
    @param wrlfile: WRL file name
    @return: coordinates
    """
    holder = []
    normholder = []  # not used
    cf = 0
    nf = 0
    with open(wrlfile, "r") as vrml:
        for lines in vrml:
            if 'point [' in lines:
                cf = 1
            if cf == 1:
                a = re.findall("[-0-9]{1,3}.[0-9]{6}", lines)
                if len(a) == 3:
                    holder.append(tuple(map(float, a)))
            if 'vector [' in lines:
                nf = 1
            if nf == 1:
                a = re.findall("[-0-9]{1}.[0-9]{4}", lines)
                if len(a) == 3:
                    normholder.append(tuple(map(float, a)))
    coords = np.array(holder)
    coords = np.unique(coords, axis=0)
    return coords


def interface_lables(path_to_receptor: str, path_to_legand: str,
                     coords_legand: np.ndarray, coords_receptor: np.ndarray, pdbid: str) -> None:
    """
    Saved Interface labels (interface points as those within 2A of a point on the partner protein)
    @param path_to_receptor: path to rf folder
    @param path_to_legand: path to lf folder
    @param coords_legand: ligand surface points P^l
    @param coords_receptor: receptor surface points P^r
    @param pdbid: structure id (four symbols)
    @return: None
    """
    assert len(pdbid) == 4

    tol = np.array([2, 2, 2])

    # all possible absolute values of distances along each coordinates
    contact = (np.abs(np.asarray(coords_legand[:, None]) - np.asarray(coords_receptor)) < tol).all(2).astype(np.int)
    llabel = np.max(contact, axis=1)
    rlabel = np.max(contact, axis=0)

    np.savetxt(f"{path_to_legand}points_label/{pdbid}-l.seg", llabel)
    np.savetxt(f"{path_to_receptor}points_label/{pdbid}-r.seg", rlabel)


def get_apbs(pdbfile: str, output_path: str, suffix: str,
             pdb2pqr: str = '~/data/PInet/pdb2pqr-linux-bin64-2.1.0/pdb2pqr',
             apbs='~/data/PInet/APBS-1.5-linux64/bin/apbs') -> None:
    """
    Preprocessed input PDB files using PDB2PQR to remove solvent molecules and fill in missing atoms and
    computed Poisson–Boltzmann electrostatics by APBS
    @param pdbfile: PDB file name
    @param output_path: output path
    @param suffix: l (legand) or r (receptor)
    @param pdb2pqr: path to the binary pdb2pqr
    @param apbs: path to the binary apbs
    @return: None
    """
    pdbid = os.path.basename(pdbfile)[:4]
    assert len(pdbid) == 4
    assert suffix in ['l', 'r']
    cmd_pdb2pqr = f"{pdb2pqr} --whitespace --ff=amber -v --apbs-input {pdbfile} {output_path}{pdbid}-{suffix}.pqr"
    cmp_apbs = f"{apbs} {output_path}{pdbid}-{suffix}.in"
    try:
        os.system(cmd_pdb2pqr)
    except:
        logging.error(f"Error when pdb2pqr l: {pdbid}")
    # add path to .pqr files in .in
    # if not os.path.isfile(f"{output_path}{pdbid}-{suffix}.in"):
    #     raise Exception
    with open(f"{output_path}{pdbid}-{suffix}.in", 'r') as f:
        data = f.read()
    data = data.replace(f"mol pqr {pdbid}-{suffix}.pqr", f"mol pqr {output_path}{pdbid}-{suffix}.pqr")
    with open(f"{output_path}{pdbid}-{suffix}.in", 'w') as f:
        f.write(data)
    try:
        os.system(cmp_apbs)
    except:
        logging.error(f"Error when abps l: {pdbid}")


def coords_to_pts(pdbfile: str, path_to_apbs: str, output_path: str, suffix: str, coords: np.ndarray) -> None:
    """
    Maybe compute hydrophobicity and something else
    @param pdbfile: PDB file name
    @param path_to_apbs: path to files from apbs (.pqr.dx files)
    @param output_path: output path (/points)
    @param suffix: l (legand) or r (receptor)
    @param coords: surface points
    @return: None
    """
    pdbid = os.path.basename(pdbfile)[:4]
    assert len(pdbid) == 4
    assert suffix in ['l', 'r']
    centroid, labels = gethydro(pdbfile)
    centroid = np.array(centroid)
    hlabel = np.transpose(np.asarray(labels[0]))
    clf = neighbors.KNeighborsClassifier(3)
    clf.fit(centroid, hlabel * 10)
    dist, ind = clf.kneighbors(coords)
    pred = np.sum(hlabel[ind] * dist, 1) / np.sum(dist, 1) / 10.0

    with open(f"{path_to_apbs}{pdbid}-{suffix}.pqr.dx", 'r') as f:
        gl, orl, dl, vl = parsefile(f)
    # print(f"pqr.dx: {path_to_apbs}{pdbid}-{suffix}.pqr.dx")
    try:
        avl = findvalue(coords, gl, orl, dl, vl)
        np.savetxt(f"{output_path}{pdbid}-{suffix}.pts",
               np.concatenate((coords, np.expand_dims(avl, 1), np.expand_dims(pred, 1)), axis=1))
    except Exception as e:
        logging.error(f"Failed: {e}!!!")


def create_folders_points(directory: str) -> None:
    """
    Create folders for legand preprocessed data and receptor preprocessed data
    @param directory: directory with legand preprocessed data and receptor preprocessed data
    @return: None
    """
    os.makedirs(os.path.join(directory, 'points'))
    os.makedirs(os.path.join(directory, 'points_label'))
    os.makedirs(os.path.join(directory, 'wrl'))
    os.makedirs(os.path.join(directory, 'pqr_in'))
    os.makedirs(os.path.join(directory, 'in'))


def ensure_dir(directory: str = "../anbase") -> None:
    """
    Ensure existence of folder for preprocessed data
    @param directory: directory for preprocessed data
    @return: None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory_lf = os.path.join(os.path.join(directory, 'lf')) # for legand
    directory_rf = os.path.join(os.path.join(directory, 'rf')) # for receptor
    if not os.path.exists(directory_lf):
        os.makedirs(directory_lf)
        create_folders_points(directory_lf)
    if not os.path.exists(directory_rf):
        os.makedirs(directory_rf)
        create_folders_points(directory_rf)


def get_pdb_files(directory: str) -> [str, str]:
    logging.info(f"Getting PDB files...")
    unbound = list()
    receptor, legand = '', ''
    for _, _, files in os.walk(directory):
        # print(f"files: {files} in {directory}")
        unbound = [f for f in files if f.endswith('_u.pdb')]
    if len(unbound) != 2:
        logging.error(f"Unbound structures in {directory}: {unbound}")
    else:
        for f in unbound:
            if f.endswith('ab_u.pdb'):
                receptor = f
            elif f.endswith('ag_u.pdb'):
                legand = f
            else:
                logging.error(f"invalid file {f}")
    logging.info(f"Got PDB files successfully!")
    return receptor, legand

def correct_pdbid(pdbid: str, pdbfile: str) -> bool:
    """
    If PDB id is correct return True, else False
    @param pdbid: PDB id
    @param pdbfile: PDB file name
    @return: True or False
    """
    if len(pdbid) != 4:
        logging.error(f"PDB id: {pdbid}, PDB file: {pdbfile}. Length of PDB id is not equal 4!")
        return False
    # print(pdbid)
    return True


def preprocess(directory: str, receptor: str, legand: str, output_path: str) -> None:
    """
    Preprocessed data
    @param directory: directo
    @param receptor:
    @param legand:
    @param output_path:
    @return:
    """
    logging.info(f"Start preprocessing...")
    pdbid = os.path.basename(receptor)[:4]
    if os.path.basename(receptor)[:4] != os.path.basename(legand)[:4]:
        logging.error(
            f"Receptor id {os.path.basename(receptor)[:4]} is not the same legand id {os.path.basename(legand)[:4]}")
    elif correct_pdbid(pdbid=pdbid, pdbfile=receptor):
        # print(f"pdbid: {pdbid}")
        path_to_receptor = os.path.join(output_path, 'rf/')
        path_to_legand = os.path.join(output_path, 'lf/')
        pdb_to_wrl(directory, receptor, pdbid, path_to_receptor, 'r')
        pdb_to_wrl(directory, legand, pdbid, path_to_legand, 'l')
        logging.info(f"Got wrl files successfully!")
        coords_receptor = get_coord(os.path.join(path_to_receptor, 'wrl', f"{pdbid}-r.wrl"))
        coords_legand = get_coord(os.path.join(path_to_legand, 'wrl', f"{pdbid}-l.wrl"))
        logging.info(f"Got interface labels successfully!")
        interface_lables(path_to_receptor, path_to_legand, coords_legand, coords_receptor, pdbid)
        get_apbs(pdbfile=os.path.join(directory, receptor), output_path=os.path.join(path_to_receptor, 'pqr_in/'),
                 suffix='r')
        get_apbs(pdbfile=os.path.join(directory, legand), output_path=os.path.join(path_to_legand, 'pqr_in/'), suffix='l')
        logging.info(f"Got apbs successfully!")
        coords_to_pts(pdbfile=os.path.join(directory, receptor), path_to_apbs=os.path.join(path_to_receptor, 'pqr_in/'),
                      output_path=os.path.join(path_to_receptor, 'points/'), suffix='r', coords=coords_receptor)
        coords_to_pts(pdbfile=os.path.join(directory, legand), path_to_apbs=os.path.join(path_to_legand, 'pqr_in/'),
                      output_path=os.path.join(path_to_legand, 'points/'), suffix='l', coords=coords_legand)
        logging.info(f"Finished preprocessing successfully!")

def get_pretrained_model(path_to_model: str = '../models/seg_model_protein_15.pth',
                         device: str = 'cpu') -> PointNetDenseCls12:
    classifier = PointNetDenseCls12(k=2, feature_transform=False, pdrop=0.0, id=5)  # default parameters
    classifier.load_state_dict(torch.load(f=path_to_model, map_location='cpu'))
    classifier.eval()
    classifier.to(device)
    return classifier

def time_info(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info(f"Execution time: {'{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)}")

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s - %(message)s', filename='preprocess_log.txt', level=logging.DEBUG)
    # path to data
    ensure_dir()
    dataset_path = '../anbase-master/data/'
    start = time.time()
    ind = 0
    complex_true = []
    complex_false = []
    for complex in tqdm(os.listdir(dataset_path)):
        ind += 1
        logging.info(f"Complex: {complex}")
        if complex in ["2nz9_F+E-B", "3g6j_H+G-D+C", "5b71_D+C-F", "2nyy_D+C-A", "6ul6_B-A",
                       "5wob_I+J-A", "6qex_C+B-A", "4gxu_U+V-I", "1e6j_H+L-P", "2vis_B+A-C",
                       "5i5k_X+Y-A", "6bf7_C+D-A"]:
            logging.error(f"This structure consumes all RAM (60Gb)!!!! Skipping!")
            complex_false.append(complex)
            continue
            # print(complex)

        # if complex == "3g6j_H+G-D+C":
        receptor, legand = get_pdb_files(os.path.join(dataset_path, complex, 'prepared_schrod/'))
        try:
            preprocess(directory=os.path.join(dataset_path, complex, 'prepared_schrod/'),
                            receptor=receptor, legand=legand, output_path='../anbase')
            complex_true.append(complex)
        except Exception as e:
            logging.error(f"Failed: {e}!!!")
            complex_false.append(complex)
            continue
    end = time.time()
    time_info(start, end)
    # torch.save(complex_true, 'complex_true')
    # torch.save(complex_false, 'complex_false')
    print('Done')
