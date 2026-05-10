sudo: stands for 'super user do'

## Directory Operations
1. `pwd`: Display present working directory
2. `mkdir mydir`: Creates a directory with the name 'mydir'
    - to create a folder structure `mkdir -p ~/test/dev/qa/prd`
3. `ls`: displays the contents of the directory
4. `ls -l`: displays the list of contents of the directory along with details like, permissions, user, created date, etc.
5. `ll`: shows the same output as `ls -l`
6. `cd <folder name>`: sets the current directory to the specified folder name. Move to the folder.
7. `~`: displays the path of home directory. `cd ~` move to parent directory
8. `cd ..`: move out of the current folder to the parent directory. `cd <abs path>` will move to the 'abs path' folder.
9. `rm -r <folder name>`: deletes folder. 


## File operations
1. `touch test.txt`: creates an empty file named test.txt
    - to create file inside a different folder: `touch ~/test/dev/qa/test.txt`
2. `echo "hello world"`: prints text on the terminal
3. `echo "hello world" > test.txt`: adds the text to the test.txt file. Adding text to file using `echo` overwrites the previous text.
4. `cat test.txt`: display the contents of the file
5. `more test.txt`: display the contents of the file page by page. If the file has 100 lines an terminal can show on 20 lines at time on screen, it will show 20 lines. To view next set of 20 lines we need to press enter.
6. `less test.txt`: opens the file on terminal and can be read line by line
7. `vi test.txt`: visual editor. Opens the file in read only mode. To edit the file we need to type 'i' (stands for insert) after opening the file. After making the changes press 'esc' to go back to the read-only mode.
    - `:wq` - write and quit. closes the file after adding the newly added text to the file
    - `:q!` - quit forcefully. closes the file without saving the newly added text
    - `:q` - quit. close the file. it will only work if the file has not been modified
    - `:wq!` - write and quit forcefully.
8. `rm test.txt`: deletes the file 'test.txt'


## Copy/Rename/Move operations
We can perform these operations using absolute or relative path, depending on the situation.

### Copy files
1. `cp a.txt b.txt`: creates a copy of a.txt and pastes it as b.txt in the same directory.
2. `cp ~/home/a.txt ~/temp/a.txt`: copies the file from 'home' to 'temp' folder
3. `cp -r ~/home/test ~/home/dev/test_new`: copies the 'test' folder in '~/home/' to '~/home/dev` with folder name 'test_new', along with the folder contents. This command performs 2 functions - copy and rename folder 

### Move folders
1. `mv config.txt dev_config.txt`: It creates a copy of config.txt and moves it to the same directory as dev_config.txt. Basically, it renames the file from config.txt to dev_config.txt because the operation is happening in the same folder.
2. `mv a.txt tmp`: moves the file in the current directory to sub-folder 'tmp' with the same name.
3. `mv test_dir new_dir`: renames the test_dir to new_dir in the current directory
4. `mv test_dir new_dir/`: moves test_dir to new_dir


## File/Directory Permissions
drwxrwxr-x or -rw-rw-r-- : denotes read/write/execute permissions of folders or files


| letters | permission | number representation |
|:-------:|:----------:|:---------------------:|
| x       | execute    | 1                     |
| r       | read       | 4                     |
| w       | write      | 2                     |
| d/-     | folder/file|                       |


1. if it start with `d` it means that the component is a directory. A file permission starts with `-`
2. the permission is combination of 10 characters. First character represents whether the permission is for a file or a directory (-/d). after that `rwx` repeated 3 times.
    - rwx + rwx + rwx: each set represents a permission level of 3 user groups - super user (root) + owner + other user
    - if any character is replaces by `-`, it means that permission is not available for the user (depending on position)
3. Taking an example permission `drwxrwxr--`
    - `d` mean that we are looking at a directory permission
    - `rwx` (characters 2-4 from left): read/write/execute permission for the super user
    - `rwx` (characters 5-7 from left): read/write/execute permission for the owner of directory
    - `r--` (last 3 characters): read permission for other users
4. `drwxr-xr-x. 2 ec2-user ec2-user 28 May 10 15:14 dev`
    - `ec2-user` first instance represents the owner of the folder
    - `ec2-user` second instance represents the group who owns the folder
5. Change the owner of the file or folder:
    `chown owner:group filename/foldername`: owner:group -> the new owner.
6. To change the permission level of files or folder use the following syntax along with sum of number representation for each group (7: rwx, 6: rw-, 5: r-x, 4: r--, 3: -wx, 2: -w-, 1: --x, 0: ---)
    - `chmod 777 file/dir` - gives rwx permission to all user groups
    - `chmod 400 file/dir` - gives read permission to the super user and no permission to owner and others. blocks any updates on the file
    - permission which number 1,2,3 means that the user will not be able to see the folder but they can write or execute (depending on permission), if they know the file/folder name

