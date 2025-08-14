# wp root folder
Adding this to python path should be enough to access all wp modules.


## Package structure
I don't pretend to know the right way, but I've found success trying to break things into similar classifications:

- **Core** packages are for deep integration with DCC, simple and atomic functions fundamental to how we work with a program, or with our code itself.
- **Lib** packages may depend on core - still atomic functions, but more free to extend. Lib files should focus on one subject, and should expand to their own subpackage at suitable size.
- **Wrapper objects** usually slot in around here, drawing logic from core and lib functions in a convenient interface - except for the WN Maya node wrappers, since those are fundamental.  
- **Util** : more like proto-tools, more high-level, far-reaching functions. Main business logic should be drawn from lib folders
- **Tool** : contained tools to manage complex processes end-to-end, maybe including scene setup, file management. As rule of thumb, if it could justify a standalone UI, it's a tool
- **Scratch** folders are designated code hellscapes - for sketches, saving gists, dirty tests with reference value enough to be versioned etc. If you ever import something live from a scratch package, check yourself.

You could certainly work out more hard rules for this - working alone, I haven't found that too useful. Sometimes there are snarls - I try and focus on fluid and intuitive imports, importing things directly from package init etc, and sometimes that means some messiness.

## File size
1000 lines is probably ok. 3000 lines is probably not.

## Naming convention 
pascalCase matches Maya, Qt and Houdini, and that's enough for me. It's also one less keypress and one character shorter to write myVar than my_var 

Lower-case file names and folders are a holdover from Framestore - they're just familiar to me. Also incentivises shorter file names.




## Versioning
On balance I prefer having this whole python system versioned in the same repo, instead of different sub-repos - we follow a principle of abstract logic specialised into software-specific uses, so unlikely different versions of packages would line up anyway.

## Constants / config files
I have absolutely no idea what the best way is to handle this. Single json file in this folder? Single wpconfig.py file at top level? Separate config files in each package? So you'll notice various flavours of this used at different places.