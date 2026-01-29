import java.io.*;
import java.util.Enumeration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

//public class getDex {
//    public static void main(String[] args) throws IOException {
//        for (int i = 1; i < 1000; i++) {
//            String zipPath = "E:\\new_data\\mal\\mal_apk1\\" + i + ".apk";
//            String outPath = "E:\\new_data\\mal\\mal_dex\\" + i;
//
//            File file = new File(zipPath);
//            if (!file.isFile()) {
//                throw new FileNotFoundException("file not exist!");
//            }
//            if (outPath == null || "".equals(outPath)) {
//                outPath = file.getParent();
//            }
//            ZipFile zipFile = new ZipFile(file);
//            Enumeration<? extends ZipEntry> files = zipFile.entries();
//            ZipEntry entry = null;
//            File outFile = null;
//            BufferedInputStream bin = null;
//            BufferedOutputStream bout = null;
//            while (files.hasMoreElements()) {
//                entry = files.nextElement();
//                outFile = new File(outPath + File.separator + entry.getName());
//                if (entry.isDirectory()) {
//                    outFile.mkdirs();
//                    continue;
//                }
//                if (!outFile.getParentFile().exists()) {
//                    outFile.getParentFile().mkdirs();
//                }
//                outFile.createNewFile();
//                if (!outFile.canWrite()) {
//                    continue;
//                }
//                try {
//                    bin = new BufferedInputStream(zipFile.getInputStream(entry));
//                    bout = new BufferedOutputStream(new FileOutputStream(outFile));
//                    byte[] buffer = new byte[1024];
//                    int readCount = -1;
//                    while ((readCount = bin.read(buffer)) != -1) {
//                        bout.write(buffer, 0, readCount);
//                    }
//                } finally {
//                    try {
//                        bin.close();
//                        bout.flush();
//                        bout.close();
//                    } catch (Exception e) {
//                    }
//                }
//            }
//            System.out.println(i + " apk complete");
//        }
//
//        for (int i = 1; i < 1000; i++) {
//            String path = "F:\\OUT\\" + i;
//            File file = new File(path);
//            File[] fs = file.listFiles();
//            for (File f : fs) {
//                if (f.isDirectory())
//                    deleteDir(f);
//                if (!f.getName().endsWith("dex"))
//                    f.delete();
//            }
//            System.out.println(i+ "  complete");
//        }
//    }
//
//    private static boolean deleteDir(File dir) {
//        if (dir.isDirectory()) {
//            String[] children = dir.list();
//            for (int i = 0; i < children.length; i++) {
//                boolean success = deleteDir(new File(dir, children[i]));
//                if (!success) {
//                    return false;
//                }
//            }
//        }
//        return dir.delete();
//    }
//}


//public class getDex {
//    public static void main(String[] args) throws IOException {
//        for (int i = 1; i < 4064; i++) {
//            String zipPath = "E:\\new_datas0-5\\mal_obf\\mal_obf_apk\\" + i + ".apk";
//            String outPath = "E:\\new_datas0-5\\mal_obf\\mal_obf_dex\\" + i;
//
//            File file = new File(zipPath);
//            if (!file.isFile()) {
//                System.err.println("File not found: " + zipPath);
//                continue;  // Skip to the next file
//            }
//
//            // 尝试打开 APK 文件
//            try (ZipFile zipFile = new ZipFile(file)) {
//                Enumeration<? extends ZipEntry> files = zipFile.entries();
//                ZipEntry entry;
//                File outFile;
//                BufferedInputStream bin = null;
//                BufferedOutputStream bout = null;
//
//                while (files.hasMoreElements()) {
//                    entry = files.nextElement();
//                    outFile = new File(outPath + File.separator + entry.getName());
//
//                    // 如果是目录，创建目录
//                    if (entry.isDirectory()) {
//                        outFile.mkdirs();
//                        continue;
//                    }
//
//                    // 确保父目录存在
//                    if (!outFile.getParentFile().exists()) {
//                        outFile.getParentFile().mkdirs();
//                    }
//
//                    outFile.createNewFile();
//
//                    // 检查输出文件是否可写
//                    if (!outFile.canWrite()) {
//                        System.err.println("Cannot write to file: " + outFile.getAbsolutePath());
//                        continue;
//                    }
//
//                    // 从 APK 中提取文件
//                    try {
//                        bin = new BufferedInputStream(zipFile.getInputStream(entry));
//                        bout = new BufferedOutputStream(new FileOutputStream(outFile));
//                        byte[] buffer = new byte[1024];
//                        int readCount;
//
//                        while ((readCount = bin.read(buffer)) != -1) {
//                            bout.write(buffer, 0, readCount);
//                        }
//                    } catch (IOException e) {
//                        System.err.println("Error extracting entry: " + entry.getName() + " - " + e.getMessage());
//                    } finally {
//                        if (bin != null) {
//                            try {
//                                bin.close();
//                            } catch (IOException e) {
//                                System.err.println("Error closing input stream: " + e.getMessage());
//                            }
//                        }
//                        if (bout != null) {
//                            try {
//                                bout.flush();
//                                bout.close();
//                            } catch (IOException e) {
//                                System.err.println("Error closing output stream: " + e.getMessage());
//                            }
//                        }
//                    }
//                }
//                System.out.println(i + " apk complete");
//            } catch (IOException e) {
//                System.err.println("Error opening APK file: " + zipPath + " - " + e.getMessage());
//                continue;  // Skip to the next file
//            }
//        }
//
//        // 删除非 DEX 文件
//        for (int i = 1; i < 4064; i++) {
//            String path = "E:\\new_datas0-5\\mal_obf\\mal_obf_dex\\" + i;
//            File file = new File(path);
//            File[] fs = file.listFiles();
//
//            if (fs != null) {
//                for (File f : fs) {
//                    if (f.isDirectory()) {
//                        deleteDir(f);
//                    } else if (!f.getName().endsWith("dex")) {
//                        f.delete();
//                    }
//                }
//            }
//            System.out.println(i + " complete");
//        }
//    }
//
//    private static boolean deleteDir(File dir) {
//        if (dir.isDirectory()) {
//            String[] children = dir.list();
//            for (String child : children) {
//                boolean success = deleteDir(new File(dir, child));
//                if (!success) {
//                    return false;
//                }
//            }
//        }
//        return dir.delete();
//    }
//}

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;


//遍历 APK → 把 APK 解压到以 APK 名命名的目录 → 在目标根目录里只保留 .dex，删掉其他文件与空文件夹。
public class getDex {
    public static void main(String[] args) throws IOException {
        // 获取文件路径列表
        Path apkDirectory = Paths.get("E:\\raw_datas\\datas-time\\googleplay_obf\\mal\\2024");
        try (Stream<Path> paths = Files.walk(apkDirectory)) {
            paths.filter(Files::isRegularFile)  // 过滤出文件
                    .filter(path -> path.toString().endsWith(".apk"))  // 只处理 APK 文件
                    .forEach(path -> processApkFile(path));
        }

        // 删除非 DEX 文件并删除空文件夹
        Path dexDirectory = Paths.get("E:\\raw_datas\\datas-time\\googleplay_obf\\mal_dex\\2024");
        try (Stream<Path> paths = Files.walk(dexDirectory)) {
            paths.filter(Files::isRegularFile)
                    .filter(path -> !path.toString().endsWith(".dex"))
                    .forEach(path -> path.toFile().delete());  // 删除非 .dex 文件
        }

        // 删除空文件夹及其内容
        deleteDirectoryRecursively(dexDirectory);

        System.out.println("处理完毕！");
    }

    // 处理 APK 文件的逻辑
    private static void processApkFile(Path apkPath) {
        String outPath = "E:\\raw_datas\\datas-time\\googleplay_obf\\mal_dex\\2024\\" + apkPath.getFileName().toString().replace(".apk", "");
        File file = apkPath.toFile();

        // 尝试打开 APK 文件
        try (ZipFile zipFile = new ZipFile(file)) {
            Enumeration<? extends ZipEntry> files = zipFile.entries();
            while (files.hasMoreElements()) {
                ZipEntry entry = files.nextElement();
                File outFile = new File(outPath + File.separator + entry.getName());

                if (entry.isDirectory()) {
                    outFile.mkdirs();
                    continue;
                }

                if (!outFile.getParentFile().exists()) {
                    outFile.getParentFile().mkdirs();
                }

                outFile.createNewFile();

                // 从 APK 中提取文件
                try (BufferedInputStream bin = new BufferedInputStream(zipFile.getInputStream(entry));
                     BufferedOutputStream bout = new BufferedOutputStream(new FileOutputStream(outFile))) {
                    byte[] buffer = new byte[1024];
                    int readCount;
                    while ((readCount = bin.read(buffer)) != -1) {
                        bout.write(buffer, 0, readCount);
                    }
                } catch (IOException e) {
                    System.err.println("Error extracting entry: " + entry.getName() + " - " + e.getMessage());
                }
            }
            System.out.println(apkPath.getFileName() + " APK complete");
        } catch (IOException e) {
            System.err.println("Error opening APK file: " + apkPath + " - " + e.getMessage());
        }
    }

    // 删除文件夹及其内容，递归删除
    private static void deleteDirectoryRecursively(Path dir) throws IOException {
        try (Stream<Path> paths = Files.walk(dir)) {
            paths.sorted(Comparator.reverseOrder())  // 先删除子文件夹，后删除父文件夹
                    .forEach(path -> {
                        try {
                            if (Files.isDirectory(path)) {
                                // 如果是目录并且目录不为空（即包含 .dex 文件），则跳过删除
                                if (Files.list(path).count() > 0) {
                                    return;  // 跳过删除包含文件的文件夹
                                }

                                // 删除空目录
                                boolean deleted = Files.deleteIfExists(path);  // 删除空文件夹
                                if (deleted) {
                                    System.out.println("删除文件夹: " + path);
                                } else {
                                    System.err.println("无法删除文件夹: " + path);
                                }
                            } else {
                                // 删除非 .dex 文件
                                if (!path.toString().endsWith(".dex")) {
                                    boolean deleted = path.toFile().delete();  // 删除非 .dex 文件
                                    if (deleted) {
                                        System.out.println("删除文件: " + path);
                                    } else {
                                        System.err.println("无法删除文件: " + path);
                                    }
                                }
                            }
                        } catch (IOException e) {
                            System.err.println("Error deleting directory or file: " + path + " - " + e.getMessage());
                        }
                    });
        }
    }
}