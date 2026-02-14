

from astropy.io import fits

def validate_hdul(hdul: fits.HDUList, required_keywords: list = None):
    """
    Validates and fixes common issues in a FITS file.
    
    Parameters:
    - hdul: HDUList: The FITS file to validate.
    - required_keywords: list: List of required keywords to check for existence.
    
    Returns:
    - None: The function will modify the FITS file in place.
    """
    # Open the FITS file

    header = hdul[0].header

    # if ('OBS NAME' in hdul[0].header):
    #     hdul[0].header['OBS-NAME'] = hdul[0].header['OBS NAME']
    #     del hdul[0].header['OBS NAME']
    
    # Check for valid keyword names and fix them
    for key in list(header.keys()):
        # Check and correct the keyword name length
        if len(key) > 8:
            # print(f"Keyword '{key}' is too long. Replacing with '{key[:8]}'")
            value = header[key]
            header.remove(key)
            header[key[:8]] = value
        
        # Check if the keyword has illegal characters or spaces
        if any(char not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_' for char in key):
            fixed_key = key.replace(' ', '_')[:8]  # Replace spaces with underscores and limit length
            # print(f"Keyword '{key}' is invalid. Renaming to '{fixed_key}'")
            
            if key in header:
                value = header[key]
                header.remove(key)
                header[fixed_key] = value
    
    # Check for duplicate keywords and remove duplicates
    # seen_keys = set()
    # duplicates = []
    # for key in header.keys():
    #     if key in seen_keys:
    #         duplicates.append(key)
    #     else:
    #         seen_keys.add(key)
    
    # for dup in duplicates:
    #     print(f"Removing duplicate keyword '{dup}'")
    #     header.remove(dup)
    
    # Check for required keywords
    if required_keywords:
        for req_key in required_keywords:
            if req_key not in header:
                # print(f"Required keyword '{req_key}' not found. Adding default value.")
                # You can set a default value if necessary
                header[req_key] = 'DEFAULT'
    
    # Check the value of the EXTEND keyword
    if 'EXTEND' in header:
        extend_value = header['EXTEND']
        
        if extend_value is None or extend_value not in [True, False, 'T', 'F']:
            # print(f"Invalid EXTEND value: {extend_value}. Setting EXTEND to True.")
            header['EXTEND'] = True  # Set to True if extensions exist
        
    else:
        # print("Adding EXTEND keyword with value True.")
        header['EXTEND'] = True  # Add the keyword if it doesn't exist
    
    # Check and fix problematic keywords
    for key in ['CD001001', 'CD002002', 'CALIBCON']: # just delete for now
        if key in header:
            value = header[key]
            print(f"Checking {key}: [{value}]")


            del header[key]

            # # Split the value to separate the numeric part from comments
            # if isinstance(value, str) and '/' in value:
            #     try:
            #         # Extract numeric value before the comment
            #         numeric_value = value.split()[0]  # Take the first part
            #         # Reassign the cleaned-up value without the comment
            #         header[key] = float(numeric_value)
            #         print(f"Fixed {key}: {header[key]}")
            #     except ValueError:
            #         print(f"Could not convert value '{value}' for {key}. Keeping the original value.")

    return hdul