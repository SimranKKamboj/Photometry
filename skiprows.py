import pickle

def get_skiprows(nights, frames_per_night, total_frames):
    skiprows_list = []
    start = 1
    for i in range(nights):
        end = start + frames_per_night[i] - 1
        if i == 0:
            skiprows = list(range(end + 1, total_frames + 1))
        elif i == nights - 1:
            skiprows = list(range(1, skiprows_list[i-1][-1] - frames_per_night[-1] + 2))
        else:
            skiprows = list(range(1, start)) + list(range(end + 2, total_frames + 1))
        skiprows_list.append(skiprows)
        start = end + 1
    return skiprows_list

def save_default_values(nights, frames_per_night, total_frames):
    with open('default_values.pkl', 'wb') as f:
        pickle.dump((nights, frames_per_night, total_frames), f)

def load_default_values():
    try:
        with open('default_values.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def main():
    default_values = load_default_values()
    if default_values:
        change_default = input("Do you want to change the default pickle file? (Y/N): ").strip().lower()
        if change_default == 'y':
            nights = int(input("Enter number of nights: "))
            frames_per_night = []
            total_frames = 0
            for i in range(nights):
                frames = int(input(f"Enter frames for night {i+1}: "))
                frames_per_night.append(frames)
                total_frames += frames
            total_frames += 1  # accounting for the header
            save_default_values(nights, frames_per_night, total_frames)
        else:
            nights, frames_per_night, total_frames = default_values
    else:
        nights = int(input("Enter number of nights: "))
        frames_per_night = []
        total_frames = 0
        for i in range(nights):
            frames = int(input(f"Enter frames for night {i+1}: "))
            frames_per_night.append(frames)
            total_frames += frames
        total_frames += 1  # accounting for the header
        save_default_values(nights, frames_per_night, total_frames)

    skiprows_list = get_skiprows(nights, frames_per_night, total_frames)
    skiprows_temp = []
    for i, skiprows in enumerate(skiprows_list):
        if i == 0:
            skiprows_temp.append(f"(list(range({skiprows[0]},{skiprows[-1] + 2})))")
        elif i == nights - 1:
            skiprows_temp.append(f"(list(range(1,{skiprows_list[i-1][-1] - frames_per_night[-1] + 1})))")
        else:
            skiprows_temp.append(f"(list(range(1,{sum(frames_per_night[:i])+1}))+list(range({sum(frames_per_night[:i+1]) + 1},{total_frames + 2})))")

    night_choice = input("Enter the night number for which you want skiprows (type 'all' for all nights): ")
    if night_choice.lower() == "all":
        output = "[]"
    else:
        night_choice = int(night_choice)
        output = skiprows_temp[night_choice - 1]

    with open('skiprows.txt', 'w') as f:
        f.write(output)

if __name__ == "__main__":
    main()
