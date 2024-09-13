def nearest_interpolation(img, new_height, new_width):
    h, w, c = img.shape
    sh = new_height / h
    sw = new_width / w
    new_size = (int(h * sh), int(w * sw))
    new_height = new_size[0] if new_height is None else min(new_height, new_size[0])
    new_width = new_size[1] if new_width is None else min(new_width, new_size[1])
    emptyImage = np.zeros((new_height, new_width, c), np.uint8)
    for i in range(new_height):
        for j in range(new_width):
            x = int((i + 0.5) / sh)
            y = int((j + 0.5) / sw)
            emptyImage[i,j]=img[x,y]
    return emptyImage

def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)
    scale_x, scale_y = src_w / dst_w, src_h / dst_h
    for dst_y in range(dst_h):
        for dst_x in range(dst_w):
            src_x = (dst_x + 0.5) * scale_x - 0.5
            src_y = (dst_y + 0.5) * scale_y - 0.5
            src_x0 = int(np.floor(src_x))
            src_x1 = min(src_x0 + 1, src_w - 1)
            src_y0 = int(np.floor(src_y))
            src_y1 = min(src_y0 + 1, src_h - 1)
            a = (src_x1 - src_x) * img[src_y0, src_x0] + (src_x - src_x0) * img[src_y0, src_x1]
            b = (src_x1 - src_x) * img[src_y1, src_x0] + (src_x - src_x0) * img[src_y1, src_x1]
            dst_img[dst_y, dst_x] = (src_y1 - src_y) * a + (src_y - src_y0) * b
    return dst_img


def histogram_equalization(image_path, display=True, save_path=None):
    img = cv2.imread(image_path, 1)
    if img is None:
        raise ValueError(f"Could not open or find the image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(gray)
    hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
    if display:
        plt.figure()
        plt.hist(dst.ravel(), 256, color='gray')
        plt.title("Histogram of Equalized Image")
        plt.xlabel("Pixel Values")
        plt.ylabel("Frequency")
        plt.show()
        cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save_path:
        cv2.imwrite(save_path, dst)
    return dst, hist
