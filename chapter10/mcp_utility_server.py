import json
import datetime
from typing import Dict, List
from fastmcp import FastMCP

# 初始化MCP服务器
mcp = FastMCP("mcp_utility_server")

# 模拟的企业数据
inventory = {
    "LAPTOP-001": {"name": "商务笔记本", "stock": 15, "price": 5999.00, "supplier": "供应商A"},
    "MOUSE-001": {"name": "无线鼠标", "stock": 50, "price": 99.00, "supplier": "供应商B"},
    "KEYBOARD-001": {"name": "机械键盘", "stock": 5, "price": 299.00, "supplier": "供应商A"},
    "MONITOR-001": {"name": "27寸显示器", "stock": 2, "price": 1299.00, "supplier": "供应商C"}
}

orders = {}
purchase_requests = {}
shipping_orders = {}
notifications = []


# 资源：获取库存信息
@mcp.resource("inventory://all")
def get_all_inventory() -> str:
    """获取所有产品的库存信息"""
    return json.dumps(inventory, ensure_ascii=False, indent=2)


@mcp.resource("inventory://{product_id}")
def get_product_inventory(product_id: str) -> str:
    """获取特定产品的库存信息"""
    if product_id in inventory:
        return json.dumps(inventory[product_id], ensure_ascii=False, indent=2)
    return json.dumps({"error": f"产品ID {product_id} 不存在"}, ensure_ascii=False)


# 工具：创建销售订单
@mcp.tool()
def create_sales_order(order_id: str, customer_id: str, items: List[Dict[str, any]]) -> str:
    """
    创建销售订单

    Args:
        order_id: 订单唯一标识
        customer_id: 客户ID
        items: 订单项目列表，每个项目包含product_id和quantity

    Returns:
        创建结果信息
    """
    # 检查订单是否已存在
    if order_id in orders:
        return json.dumps({"error": f"订单 {order_id} 已存在"}, ensure_ascii=False)

    # 验证产品ID和数量
    for item in items:
        product_id = item.get("product_id")
        quantity = item.get("quantity", 0)

        if product_id not in inventory:
            return json.dumps({"error": f"产品ID {product_id} 不存在"}, ensure_ascii=False)

        if quantity <= 0:
            return json.dumps({"error": f"产品 {product_id} 的数量必须大于0"}, ensure_ascii=False)

    # 创建订单
    order = {
        "order_id": order_id,
        "customer_id": customer_id,
        "items": items,
        "status": "PENDING",
        "created_at": datetime.datetime.now().isoformat(),
        "total_amount": 0
    }

    # 计算总金额
    total = 0
    for item in items:
        product_id = item.get("product_id")
        quantity = item.get("quantity")
        total += inventory[product_id]["price"] * quantity

    order["total_amount"] = total
    orders[order_id] = order

    return json.dumps({"success": True, "order": order}, ensure_ascii=False, indent=2)


# 工具：检查库存并创建采购申请
@mcp.tool()
def check_inventory_and_create_purchase_request(order_id: str) -> str:
    """
    检查订单所需库存，如果库存不足则创建采购申请

    Args:
        order_id: 订单ID

    Returns:
        检查结果和可能的采购申请信息
    """
    if order_id not in orders:
        return json.dumps({"error": f"订单 {order_id} 不存在"}, ensure_ascii=False)

    order = orders[order_id]
    items_to_purchase = []

    # 检查每个项目的库存
    for item in order["items"]:
        product_id = item["product_id"]
        required_quantity = item["quantity"]
        available_stock = inventory[product_id]["stock"]

        if available_stock < required_quantity:
            items_to_purchase.append({
                "product_id": product_id,
                "product_name": inventory[product_id]["name"],
                "required_quantity": required_quantity,
                "available_stock": available_stock,
                "shortfall": required_quantity - available_stock,
                "supplier": inventory[product_id]["supplier"]
            })

    # 如果需要采购
    if items_to_purchase:
        pr_id = f"PR-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        purchase_request = {
            "pr_id": pr_id,
            "order_id": order_id,
            "items": items_to_purchase,
            "status": "PENDING_APPROVAL",
            "created_at": datetime.datetime.now().isoformat()
        }
        purchase_requests[pr_id] = purchase_request

        return json.dumps({
            "sufficient_stock": False,
            "purchase_request": purchase_request,
            "message": f"库存不足，已创建采购申请 {pr_id}"
        }, ensure_ascii=False, indent=2)

    return json.dumps({
        "sufficient_stock": True,
        "message": "所有产品库存充足，可以继续处理订单"
    }, ensure_ascii=False, indent=2)


# 工具：批准采购申请
@mcp.tool()
def approve_purchase_request(pr_id: str, approved_by: str) -> str:
    """
    批准采购申请

    Args:
        pr_id: 采购申请ID
        approved_by: 批准人

    Returns:
        批准结果
    """
    if pr_id not in purchase_requests:
        return json.dumps({"error": f"采购申请 {pr_id} 不存在"}, ensure_ascii=False)

    purchase_request = purchase_requests[pr_id]
    purchase_request["status"] = "APPROVED"
    purchase_request["approved_by"] = approved_by
    purchase_request["approved_at"] = datetime.datetime.now().isoformat()

    # 模拟采购后库存增加
    for item in purchase_request["items"]:
        product_id = item["product_id"]
        shortfall = item["shortfall"]
        inventory[product_id]["stock"] += shortfall

    return json.dumps({
        "success": True,
        "purchase_request": purchase_request,
        "message": f"采购申请 {pr_id} 已批准，库存已更新"
    }, ensure_ascii=False, indent=2)


# 工具：创建发货单
@mcp.tool()
def create_shipping_order(order_id: str) -> str:
    """
    为订单创建发货单

    Args:
        order_id: 订单ID

    Returns:
        发货单信息
    """
    if order_id not in orders:
        return json.dumps({"error": f"订单 {order_id} 不存在"}, ensure_ascii=False)

    order = orders[order_id]

    # 检查库存是否充足
    for item in order["items"]:
        product_id = item["product_id"]
        required_quantity = item["quantity"]
        available_stock = inventory[product_id]["stock"]

        if available_stock < required_quantity:
            return json.dumps({
                "error": f"产品 {product_id} 库存不足，无法创建发货单",
                "available_stock": available_stock,
                "required_quantity": required_quantity
            }, ensure_ascii=False)

    # 创建发货单
    shipping_id = f"SH-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    shipping_order = {
        "shipping_id": shipping_id,
        "order_id": order_id,
        "customer_id": order["customer_id"],
        "items": order["items"].copy(),
        "status": "READY_TO_SHIP",
        "created_at": datetime.datetime.now().isoformat()
    }

    shipping_orders[shipping_id] = shipping_order

    # 更新库存
    for item in order["items"]:
        product_id = item["product_id"]
        quantity = item["quantity"]
        inventory[product_id]["stock"] -= quantity

    # 更新订单状态
    order["status"] = "SHIPPED"

    return json.dumps({
        "success": True,
        "shipping_order": shipping_order,
        "message": f"已为订单 {order_id} 创建发货单 {shipping_id}"
    }, ensure_ascii=False, indent=2)


# 工具：发送通知
@mcp.tool()
def send_notification(recipient: str, message: str, notification_type: str = "INFO") -> str:
    """
    发送通知给相关人员

    Args:
        recipient: 接收人
        message: 通知内容
        notification_type: 通知类型 (INFO, WARNING, ERROR)

    Returns:
        通知发送结果
    """
    notification = {
        "id": len(notifications) + 1,
        "recipient": recipient,
        "message": message,
        "type": notification_type,
        "sent_at": datetime.datetime.now().isoformat()
    }

    notifications.append(notification)

    return json.dumps({
        "success": True,
        "notification": notification,
        "message": f"已向 {recipient} 发送 {notification_type} 通知"
    }, ensure_ascii=False, indent=2)


# 工具：获取订单状态
@mcp.tool()
def get_order_status(order_id: str) -> str:
    """
    获取订单的当前状态

    Args:
        order_id: 订单ID

    Returns:
        订单状态信息
    """
    if order_id not in orders:
        return json.dumps({"error": f"订单 {order_id} 不存在"}, ensure_ascii=False)

    order = orders[order_id]

    # 检查是否有相关的采购申请
    related_pr = None
    for pr_id, pr in purchase_requests.items():
        if pr["order_id"] == order_id:
            related_pr = pr
            break

    # 检查是否有发货单
    related_shipping = None
    for shipping_id, shipping in shipping_orders.items():
        if shipping["order_id"] == order_id:
            related_shipping = shipping
            break

    result = {
        "order": order,
        "purchase_request": related_pr,
        "shipping_order": related_shipping
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


# 启动服务器
if __name__ == "__main__":
    mcp.run()