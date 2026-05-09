---
title: "Hugo + Even 主题搭建个人博客"
date: 2026-04-26
lastmod: 2026-04-26
draft: false
keywords: ["Hugo", "博客", "Even"]
description: "从零搭建 Hugo 博客的完整记录"
tags: ["Hugo", "建站"]
categories: ["技术"]
toc: true
mathjax: false
---

记录一下搭建这个博客的过程，方便以后参考，也给有同样需求的人一个参考。

<!--more-->

## 为什么选择 Hugo

- 构建速度极快，毫秒级
- Go 语言编写，单二进制文件，无依赖
- 主题生态丰富

## 为什么选择 Even 主题

- 简洁干净，阅读体验好
- 原生支持中文
- 配置项丰富但不复杂

## 基本用法

写文章只需要在 `content/post/` 目录下新建一个 `.md` 文件：

```bash
hugo new post/my-new-post.md
```

本地预览：

```bash
hugo server -D
```

生成静态文件：

```bash
hugo
```

生成的文件在 `public/` 目录下，部署到任意静态托管服务即可。
