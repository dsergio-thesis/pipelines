
import glob
import os
import pandas as pd


from astropy.coordinates import SkyCoord
from astropy import units as u

from abc import ABC, abstractmethod

from astroquery.simbad import Simbad

class AstroosCatalog(ABC):

    def __init__(self, dir, catalog_dir, max_records=None):
        self.catalog_dir = catalog_dir
        self.dir = dir
        self.data = pd.DataFrame()
        self.max_records = max_records

    def concat_catalog(self):

        csv_files = glob.glob(f"{self.catalog_dir}/*.csv")
        data_frames = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            data_frames.append(df)

        all_data = pd.concat(data_frames, ignore_index=True)

        # Save the combined DataFrame to a new CSV file
        # all_data.to_csv(f"{self.dir}/all.csv", index=False)

        del data_frames
        del df
        del all_data

    def load_catalog(self):
        csv_files = glob.glob(f"{self.catalog_dir}/*.csv")
        data_frames = []
        for csv_file in csv_files:

            # if csv_file ends with f"{self.pipeline.max_records}.csv", load it
            if csv_file.endswith(f"{self.max_records}.csv"):
                df = pd.read_csv(csv_file)
                data_frames.append(df)

        self.data = pd.concat(data_frames, ignore_index=True)
        self.data.reset_index(drop=True, inplace=True)
        print(f"SAVING TO self.dir: {self.dir} self.catalog_dir: {self.catalog_dir}")
        self.data.to_csv(f"{self.catalog_dir}/full_catalog_limit_{self.max_records}.csv", index=True)

        del data_frames
        del df
    
    def load_single_catalog(self):
        csv_files = glob.glob(f"{self.catalog_dir}/*.csv")
        data_frames = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            data_frames.append(df)
        self.data = pd.concat(data_frames, ignore_index=True)
        self.data.reset_index(drop=True, inplace=True)

        del data_frames
        del df

    @abstractmethod
    def filter_catalog(self):
        pass

# ============================================================
# AstroosCatalogLSST
# ============================================================
class AstroosCatalogLSST(AstroosCatalog):
    
    def __init__(self, dir, catalog_dir, pipeline, max_records=None):
        super().__init__(dir, catalog_dir=catalog_dir, max_records=max_records)
        self.catalog_dir = catalog_dir
        self.dir = dir
        self.pipeline = pipeline

    def filter_catalog(self):

        positions_filtered = []
        labels_filtered = []
        rows_filtered = []
        c = 1
        overall_count = 0
        output = ""

        print(f"Filter data length: {len(self.data)}")
        total_scan = len(self.data)

        for i in range(0, total_scan):

            print(f"[INFO] Processing {i}/{total_scan}...")

            if self.max_records is not None and c > self.max_records:
                print(f"[INFO] Reached max_records limit: {self.max_records}. Stopping.")
                break

            overall_count += 1

            if i < len(self.data):

                row = self.data.iloc[i]



                coord = SkyCoord(ra=row['coord_ra'], dec=row['coord_dec'], unit=(u.deg, u.deg))

                res = Simbad.query_region(coord, radius=5 * u.arcsec)

                if res is None or len(res) == 0:
                    print(f"[WARNING] No Simbad result for RA={row['coord_ra']}, Dec={row['coord_dec']}. Test Data.")

                    positions_filtered.append((row['coord_ra'], row['coord_dec']))
                    labels_filtered.append(999)  # unknown label
                    rows_filtered.append(row)
                    continue

                row = row.append(res[0].to_pandas().iloc[0])
                morph_type = str(row['morph_type'])

                print("row: " + str(row) + ", morph_type: " + morph_type)

                if morph_type == 'nan':
                    morph_type = ""

                label_index = self.pipeline._get_label_index(morph_type)

                output = f"[INFO] {c} / {morph_type}\t"

                if label_index is None:
                    # add to label_definitions
                    self.pipeline._add_label(morph_type)
                    label_index = self.pipeline._get_label_index(morph_type)
                    # raise ValueError(f"morph_type {row['morph_type']} not found in label definitions.")
                

                else:
                    output += f" label_index: {label_index} | "

                position = SkyCoord(row['coord_ra'], row['coord_dec'], unit=(u.deg, u.deg))
                # print("position: " + str(position))

                positions_filtered.append((position.ra.deg, position.dec.deg))
                rows_filtered.append(row)
                output += f"RA: {row['coord_ra']}, DEC: {row['coord_dec']} | "
                

                
                c += 1

                if row['otype_longname'] == 'Galaxy':
                    labels_filtered.append(label_index)
                else:
                    labels_filtered.append(999)
                
                output += f" record {i} | ID: {row['main_id']} | RA: {row['ra']}, DEC: {row['dec']} | label_index: {label_index}"
                
                j = 0


                
        
        return positions_filtered, labels_filtered, rows_filtered

# ============================================================
# AstroosCatalogSDSS
# ============================================================
class AstroosCatalogSDSS(AstroosCatalog):


    def __init__(self, dir, catalog_dir, pipeline, max_records=None):
        super().__init__(dir, catalog_dir=catalog_dir, max_records=max_records)
        self.catalog_dir = catalog_dir
        self.dir = dir
        self.pipeline = pipeline

    def filter_catalog(self):

        positions_filtered = []
        labels_filtered = []
        rows_filtered = []

        c = 1
        overall_count = 0
        
        output = ""

        print(f"Filter data length: {len(self.data)}")

        total_scan = len(self.data)

        for i in range(0, total_scan):

            print(f"[INFO] Processing {i}/{total_scan}...")

            if self.max_records is not None and c > self.max_records:
                print(f"[INFO] Reached max_records limit: {self.max_records}. Stopping.")
                break

            overall_count += 1

            if i < len(self.data):

                print_output = True

                try:

                    row = self.data.iloc[i]
                    morph_type = str(row['morph_type'])

                    # print("row: " + str(row) + ", morph_type: " + morph_type)

                    if morph_type == 'nan':
                        morph_type = ""

                    label_index = self.pipeline._get_label_index(morph_type)

                    otype_longname = row['otype_longname']

                    output = f"[INFO] {c}/ {otype_longname} {morph_type}\t"

                    include_stars = True

                    if (include_stars):
                        if (otype_longname != 'Galaxy' and otype_longname != 'Star'):
                            raise ValueError(f"otype_longname {row['otype_longname']} is not valid")
                    else:
                        if (otype_longname != 'Galaxy'):
                            raise ValueError(f"otype_longname {row['otype_longname']} is not valid")
                    
                    
                    

                    if label_index is None and otype_longname == 'Galaxy':
                        # add to label_definitions
                        self.pipeline._add_label(morph_type)
                        label_index = self.pipeline._get_label_index(morph_type)
                        # raise ValueError(f"morph_type {row['morph_type']} not found in label definitions.")
                    

                    else:
                        output += f" label_index: {label_index} | "

                    position = SkyCoord(row['ra'], row['dec'], unit=(u.deg, u.deg))
                    # print("position: " + str(position))

                    # filter by galaxy cluster bounding boxes 
                    # (ra_min, ra_max, dec_min, dec_max)
                    #
                    # coma cluster
                    coma_cluster_bounding_box = (193, 200, 25, 30) 
                    # virgo cluster
                    virgo_cluster_bounding_box = (180, 195, 5, 20) 
                    # perseus cluster
                    perseus_cluster_bounding_box = (48, 51, 39, 44)
                    # norma cluster
                    norma_cluster_bounding_box = (240, 245, -62, -58)

                    bounding_boxes = []
                    bounding_boxes.append(coma_cluster_bounding_box)
                    bounding_boxes.append(virgo_cluster_bounding_box)
                    bounding_boxes.append(perseus_cluster_bounding_box)
                    bounding_boxes.append(norma_cluster_bounding_box)

                    in_any_box = True

                    for box in bounding_boxes:
                        ra_min, ra_max, dec_min, dec_max = box

                        if (position.ra.deg >= ra_min and position.ra.deg <= ra_max) and \
                            (position.dec.deg >= dec_min or position.dec.deg <= dec_max):
                            in_any_box = True

                    if (not in_any_box):
                        raise ValueError(f"Position {position} is out of bounds for bounding box {box}.")

                    positions_filtered.append((position.ra.deg, position.dec.deg))
                    rows_filtered.append(row)
                    output += f"RA: {row['ra']}, DEC: {row['dec']} | "
                    

                    
                    c += 1

                    if row['otype_longname'] == 'Galaxy':
                        labels_filtered.append(label_index)
                    else:
                        labels_filtered.append(999)
                    
                    output += f" record {i} | ID: {row['main_id']} | RA: {row['ra']}, DEC: {row['dec']} | label_index: {label_index}"
                    
                    j = 0

                except Exception as e:
                    output += f"\n[ERROR] {row['main_id']}: {e}"
                    # print_output = False
                
                if print_output:
                        print(output)
                
        
        return positions_filtered, labels_filtered, rows_filtered
